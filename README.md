# Face Recognition API

Python + FastAPI + InsightFace ile gelistirilmis, Docker uzerinde calisan bir yuz tanima servisidir.

## Gereksinimler

- Docker (>= 24.0)
- Docker Compose (>= 2.20)

Kontrol etmek icin:

```bash
docker --version
docker compose version
```

## Hizli Baslangic

```bash
# 1. Projeyi klonla
git clone <repo-url> && cd face-recognition-api

# 2. Build et ve ayaga kaldir
docker compose up --build

# 3. Servis calisiyor mu kontrol et
curl http://localhost:8000/docs
```

## Ortam Degiskenleri (.env)

Proje kokunde bir `.env` dosyasi olusturarak tum ayarlari ozellestirebilirsiniz:

```env
# ---------- PostgreSQL ----------
POSTGRES_DB=face_db
POSTGRES_USER=face_user
POSTGRES_PASSWORD=gizli_sifre
POSTGRES_PORT=5432

# ---------- API ----------
DATABASE_URL=postgresql://face_user:gizli_sifre@db:5432/face_db
SIMILARITY_THRESHOLD=0.45
INSIGHTFACE_MODEL_NAME=buffalo_l
API_PORT=8000
```

`.env` dosyasi yoksa docker-compose.yml icindeki varsayilan degerler kullanilir.

## Docker Imajinin Yapisi

Dockerfile su asamalardan gecer:

1. **Sistem paketleri** – OpenCV icin gerekli kutuphaneler yuklenir.
2. **Python bagimliliklari** – `requirements.txt` uzerinden pip ile kurulur.
3. **Model indirme** – `download_models.py` calistirilir, InsightFace `buffalo_l` modeli
   (`./models/` altina) indirilir.
4. **Uygulama kodu** – `COPY . .` ile image'a eklenir.

Model dosyalari image'a **build asamasinda** gomuldugu icin, container basladiginda
internet baglantisi gerekmez.

## API Kullanim Kilavuzu

| Metod   | Endpoint                      | Aciklama                              |
|---------|-------------------------------|----------------------------------------|
| `POST`  | `/faces/recognize`            | Goruntudeki yuzleri tespit et/tani     |
| `POST`  | `/faces/enroll`               | Yeni yuz kaydet / anonim yuzu isimlendir |
| `GET`   | `/faces/{faceId}`             | Yuz detayini getir                     |
| `DELETE`| `/faces/{faceId}`             | Yuz kaydini sil                        |
| `GET`   | `/faces/{faceId}/history`     | Yuzun gecmis goruntulenmeleri          |
| `GET`   | `/processes/{processId}`      | Islemin detaylarini getir              |

Swagger dokumantasyonu: [http://localhost:8000/docs](http://localhost:8000/docs)

## Ornek Kullanim

```bash
# 1. Yeni bir yuz tanit (enroll)
curl -X POST http://localhost:8000/faces/enroll \
  -F "file=@ornek_fotograf.jpg" \
  -F "name=Ahmet Yilmaz" \
  -F 'metadata={"title":"Muhendis"}'

# 2. Bir goruntude yuz ara (recognize)
curl -X POST http://localhost:8000/faces/recognize \
  -F "file=@test_gorseli.jpg"

# 3. Yuz gecmisini sorgula
curl http://localhost:8000/faces/<faceId>/history

# 4. Islem detayini sorgula
curl http://localhost:8000/processes/<processId>
```

## Test

```bash
# Birim testleri (Docker gerektirmez)
cd face-recognition-api
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-mock httpx
PYTHONPATH=. pytest tests/ -v
```
