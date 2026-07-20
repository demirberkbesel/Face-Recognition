from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from app.database import engine, Base
from app.config import RATE_LIMIT
from app.rate_limit import RateLimitMiddleware
from app.routers import faces, processes

app = FastAPI(
    title="Face Recognition API",
    description="Görüntüler üzerinden yüz tanıma, kayıt ve geçmiş sorgulama servisi.",
    version="1.0.0",
)


def _parse_rate_limit(value: str) -> tuple[int, int]:
    """ '100/minute' veya '30/second' → (limit, window_seconds) """
    part = value.strip()
    if "/" in part:
        num, unit = part.split("/", 1)
        limit = int(num.strip())
        unit = unit.strip().lower()
        window = {"minute": 60, "second": 1, "hour": 3600}.get(unit, 60)
    else:
        limit = int(part)
        window = 60
    return limit, window


_rate_limit, _rate_window = _parse_rate_limit(RATE_LIMIT)

app.add_middleware(
    RateLimitMiddleware,
    limit=_rate_limit,
    window=_rate_window,
)


@app.on_event("startup")
def on_startup():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_face_embeddings_hnsw "
            "ON face_embeddings USING hnsw (embedding vector_cosine_ops)"
        ))
        conn.commit()


app.include_router(faces.router)
app.include_router(processes.router)

# En alta yazılır ki /docs, /faces/*, /processes/* öncelikli kalsın
app.mount("/", StaticFiles(directory="fe", html=True), name="fe")
