from fastapi import FastAPI

from app.database import engine, Base
from app.routers import faces, processes

app = FastAPI(
    title="Face Recognition API",
    description="Görüntüler üzerinden yüz tanıma, kayıt ve geçmiş sorgulama servisi.",
    version="1.0.0",
)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


app.include_router(faces.router)
app.include_router(processes.router)
