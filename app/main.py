from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from app.database import engine, Base
from app.routers import faces, processes

app = FastAPI(
    title="Face Recognition API",
    description="Görüntüler üzerinden yüz tanıma, kayıt ve geçmiş sorgulama servisi.",
    version="1.0.0",
)


@app.on_event("startup")
def on_startup():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(bind=engine)


app.include_router(faces.router)
app.include_router(processes.router)

# En alta yazılır ki /docs, /faces/*, /processes/* öncelikli kalsın
app.mount("/", StaticFiles(directory="fe", html=True), name="fe")
