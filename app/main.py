from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.database import engine, Base
from app.limiter import limiter
from app.routers import faces, processes

app = FastAPI(
    title="Face Recognition API",
    description="Görüntüler üzerinden yüz tanıma, kayıt ve geçmiş sorgulama servisi.",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


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
