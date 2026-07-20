# Face Recognition API

Gönderilen bir görseldeki tüm yüzleri tespit edip tanıyan, REST API olarak çalışan bir yüz tanıma servisidir. Basit bir web arayüzü `http://localhost:8000/` adresinde sunulur, isteğe bağlı olarak doğrudan API de kullanılabilir.

## Kullanılan Teknolojiler

| Bileşen | Teknoloji |
|---|---|
| Web çatısı | **FastAPI** (Python 3.12) |
| Yüz tespiti ve tanıma | **InsightFace** (`buffalo_l`, ONNX Runtime, CPU) |
| Veritabanı | **PostgreSQL 16** + **pgvector** (vektör benzerlik arama) |
| ORM | **SQLAlchemy 2.0** |
| Test | **pytest**, **pytest-mock**, **httpx** (TestClient) |
| Container | **Docker** + **docker-compose** |

InsightFace model dosyaları (`buffalo_l`) Docker imajının build aşamasında önceden indirilip imaja gömülür. Container başlatıldığında internet bağlantısı gerekmez, çevrimdışı çalışır.

## Gereksinimler

- Docker (>= 24.0)
- Docker Compose (>= 2.20)

Kontrol etmek için:
```bash
docker --version
docker compose version
```

## Çalıştırma

```bash
# Projeyi klonla
git clone <repo-url>
cd face-recognition-api

# Build et ve ayağa kaldır
docker compose up --build
```

İlk çalıştırmada InsightFace model dosyaları (`~150MB`) indirilip imaja gömüldüğü için build birkaç dakika sürebilir. Sonraki çalıştırmalarda `--build` kullanmadan doğrudan `docker compose up` ile başlatabilirsiniz.

API adresi: **http://localhost:8000**  
Web arayüzü: **http://localhost:8000** (ana sayfa)  
Swagger dokümantasyonu: **http://localhost:8000/docs**

### Durdurma

```bash
# Container'ları durdur (veritabanı ve fotoğraflar kalır)
docker compose down

# Container'ları durdur ve tüm verileri sil (volume dahil)
docker compose down -v
```

## Yapılandırma

Tüm ayarlar ortam değişkenleri (environment variables) ile yapılır. Proje kökünde bir `.env` dosyası oluşturarak varsayılan değerleri değiştirebilirsiniz.

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `POSTGRES_DB` | `face_db` | PostgreSQL veritabanı adı |
| `POSTGRES_USER` | `face_user` | PostgreSQL kullanıcı adı |
| `POSTGRES_PASSWORD` | `face_password` | PostgreSQL şifresi |
| `POSTGRES_PORT` | `5433` | Host üzerinde PostgreSQL bağlantı portu (container içi 5432) |
| `DATABASE_URL` | `postgresql://face_user:face_password@db:5432/face_db` | Veritabanı bağlantı dizesi (`postgresql://kullanici:sifre@host:port/db` formatında) |
| `SIMILARITY_THRESHOLD` | `0.45` | Kosinüs mesafesi eşiğini belirler. Yüksek değer daha katı eşleşme (daha az hatalı pozitif), düşük değer daha toleranslı eşleşme (daha az hatalı negatif) demektir. |
| `MIN_DETECTION_CONFIDENCE` | `0.5` | InsightFace yüz tespit güven skoru eşiği. Altındaki yüzler `qualityWarning` ile işaretlenir ve embedding'i kaydedilmez (yanlış eşleşmeyi önler). |
| `BLUR_THRESHOLD` | `10.0` | Laplacian varyans bulanıklık eşiği. Altındaki görseller bulanık kabul edilip reddedilir. Düşük değer = daha toleranslı. |
| `INSIGHTFACE_MODEL_NAME` | `buffalo_l` | InsightFace model seti adı |
| `API_PORT` | `8000` | Host üzerinde API portu |

## API Endpoint'leri

| Metot | Path | Açıklama |
|---|---|---|
| `POST` | `/faces/recognize` | Görseldeki tüm yüzleri tespit eder, tanır ve sonucu döner |
| `POST` | `/faces/enroll` | Yeni bir yüz kaydeder veya anonim bir yüzü isimlendirir |
| `GET` | `/faces/{face_id}` | Bir yüz kimliğinin detaylarını döner |
| `GET` | `/faces/{face_id}/image` | Kırpılmış yüz resmini `image/jpeg` olarak döndürür |
| `DELETE` | `/faces/{face_id}` | Bir yüz kimliğini ve bağlı tüm verileri siler |
| `GET` | `/faces/{face_id}/history` | Bir yüzün geçmişte hangi işlemlerde göründüğünü döner |
| `GET` | `/processes/{process_id}` | Bir işlemin detaylarını ve o işlemde tespit edilen yüzleri döner |

### POST /faces/recognize

Görseldeki yüzleri tespit eder, varsa kayıtlı kimlikle eşleştirir, yoksa yeni anonim kayıt oluşturur.

**Request:** `multipart/form-data`
- `file` (File) — JPEG veya PNG formatında görsel dosyası

**Örnek:**
```bash
curl -X POST http://localhost:8000/faces/recognize \
  -F "file=@kisi.jpg"
```

**Başarılı yanıt (yeni anonim yüz):**
```json
{
  "processId": "5acdb121-6743-4f35-abee-5ce756fe71f6",
  "faceCount": 1,
  "faces": [
    {
      "faceId": "04e57368-bbdd-4f91-be5b-9b50b9dd5e30",
      "status": "new_anonymous",
      "name": null,
      "metadata": null,
      "boundingBox": {
        "xmin": 69,
        "ymin": 60,
        "xmax": 170,
        "ymax": 195
      },
      "confidence": 0.0
    }
  ]
}
```

**Başarılı yanıt (yüz bulunamadı):**
```json
{
  "processId": "5acdb121-6743-4f35-abee-5ce756fe71f6",
  "faceCount": 0,
  "faces": []
}
```

### POST /faces/enroll

**İki kullanım şekli vardır:**

#### 1. Var olan anonim bir yüzü isimlendirme

Önce `/faces/recognize` ile `faceId` alınır, sonra bu endpoint ile isim verilir.

**Request:** `multipart/form-data`
- `face_id` (string) — İsimlendirilecek yüz kimliği
- `name` (string) — Kişi adı
- `metadata` (string, opsiyonel) — JSON formatında ek bilgiler

**Örnek:**
```bash
curl -X POST http://localhost:8000/faces/enroll \
  -F "face_id=47cb324c-1e24-4f9e-bd9d-bc11d290b0e5" \
  -F "name=Mehmet Demir" \
  -F 'metadata={"role":"Tasarımcı"}'
```

**Başarılı yanıt:**
```json
{
  "faceId": "04e57368-bbdd-4f91-be5b-9b50b9dd5e30",
  "status": "known",
  "name": "Abdullah Gul",
  "metadata": null
}
```

#### 2. Yeni bir yüzü doğrudan kaydetme (görsel ile)

**Request:** `multipart/form-data`
- `file` (File) — Yüzü içeren görsel
- `name` (string) — Kişi adı
- `metadata` (string, opsiyonel) — JSON formatında ek bilgiler

**Örnek:**
```bash
curl -X POST http://localhost:8000/faces/enroll \
  -F "file=@kisi.jpg" \
  -F "name=Ahmet Yılmaz" \
  -F 'metadata={"title":"Mühendis"}'
```

### GET /faces/{face_id}

Bir yüz kimliğinin detaylarını döner. Yanıtta o kişiye ait en son kırpılmış yüz fotoğrafının yolu da (`imagePath`) yer alır.

```bash
curl http://localhost:8000/faces/04e57368-bbdd-4f91-be5b-9b50b9dd5e30
```

```json
{
  "faceId": "04e57368-bbdd-4f91-be5b-9b50b9dd5e30",
  "status": "known",
  "name": "Abdullah Gul",
  "metadata": null,
  "imagePath": "/faces/04e57368-bbdd-4f91-be5b-9b50b9dd5e30/image"
}
```

### DELETE /faces/{face_id}

Bir yüz kimliğini ve ona bağlı tüm embedding vektörlerini, geçmiş kayıtlarını ve kırpılmış fotoğraflarını siler.

```bash
curl -X DELETE http://localhost:8000/faces/04e57368-bbdd-4f91-be5b-9b50b9dd5e30
```

```json
{
  "message": "Face ID ve bağlı vektörler başarıyla silindi.",
  "faceId": "04e57368-bbdd-4f91-be5b-9b50b9dd5e30"
}
```

### GET /faces/{face_id}/history

Bir yüz kimliğinin geçmiş işlem kayıtlarını döner.

```bash
curl http://localhost:8000/faces/04e57368-bbdd-4f91-be5b-9b50b9dd5e30/history
```

```json
{
  "faceId": "04e57368-bbdd-4f91-be5b-9b50b9dd5e30",
  "history": [
    {
      "processId": "88f21775-25fc-4a42-a825-c65e7e428d3b",
      "timestamp": "2026-07-13T12:13:30.251490",
      "status": "known",
      "confidence": 0.64,
      "boundingBox": {
        "xmin": 74, "ymin": 66, "xmax": 174, "ymax": 191
      }
    },
    {
      "processId": "5acdb121-6743-4f35-abee-5ce756fe71f6",
      "timestamp": "2026-07-13T12:13:28.817992",
      "status": "new_anonymous",
      "confidence": 0.0,
      "boundingBox": {
        "xmin": 69, "ymin": 60, "xmax": 170, "ymax": 195
      }
    }
  ]
}
```

### GET /processes/{process_id}

Tek bir işlemin detaylarını ve o işlemde tespit edilen yüzleri döner.

```bash
curl http://localhost:8000/processes/5acdb121-6743-4f35-abee-5ce756fe71f6
```

```json
{
  "processId": "5acdb121-6743-4f35-abee-5ce756fe71f6",
  "timestamp": "2026-07-13T12:13:28.817992",
  "taskDetails": {
    "type": "recognition",
    "faceCount": 1
  },
  "faces": [
    {
      "faceId": "04e57368-bbdd-4f91-be5b-9b50b9dd5e30",
      "status": "new_anonymous",
      "confidence": 0.0,
      "boundingBox": {
        "xmin": 69, "ymin": 60, "xmax": 170, "ymax": 195
      }
    }
  ]
}
```

## Kimlik Durumları (Status)

Her yüz bir `status` değerine sahiptir:

| Değer | Anlamı |
|---|---|
| `known` | Sistemde kayıtlı, ismi ve metadata'sı olan yüz. `/faces/enroll` ile isimlendirilmiş demektir. |
| `anonymous` | Daha önce görülmüş, sisteme kaydedilmiş ancak henüz isimlendirilmemiş yüz. Herhangi bir kişisel bilgi içermez. |
| `new_anonymous` | Bu istek sırasında **ilk kez** görülen yüz. Sistem tarafından otomatik oluşturulmuş yeni anonim kayıttır. Sonraki isteklerde aynı yüz gelirse `anonymous` olarak tanınır. |

`known` durumunda `name` ve `metadata` alanları dolu olur. `anonymous` ve `new_anonymous` durumlarında `name` ve `metadata` her zaman `null`'dur.

## Fotoğraf Saklama

Her `/faces/recognize` çağrısında, tespit edilen her yüz, InsightFace'in verdiği `boundingBox` koordinatları kullanılarak orijinal görselden kırpılır ve `/app/images/{identity_id}/{uuid}.jpg` yoluna kaydedilir. Bu dosyalar Docker volume'u (`images_data`) üzerinde saklandığı için container yeniden başlatıldığında kaybolmaz. Dosya yolu, `face_embeddings` tablosundaki `image_path` sütununda tutulur.

**Kırpılmış resme erişim:** `GET /faces/{face_id}/image` endpoint'i doğrudan `image/jpeg` olarak resmi döndürür. `GET /faces/{face_id}` cevabındaki `imagePath` alanı da bu URL'i (`/faces/{face_id}/image`) gösterir.

## Testler

### Birim Testleri (pytest)

Veritabanı ve InsightFace mock'lanarak çalışır, Docker gerektirmez.

```bash
# Sanal ortam oluştur
python3 -m venv venv
source venv/bin/activate

# Bağımlılıkları yükle
pip install -r requirements.txt
pip install pytest pytest-mock httpx

# Testleri çalıştır
PYTHONPATH=. pytest tests/ -v
```

29 test senaryosu:
- Yeni anonim yüz oluşturma
- Bilinen (known) yüz eşleşmesi
- Anonim yüz eşleşmesi
- Eşik altı benzerlik → yeni anonim
- Aynı görselde birden çok yüz
- Görselle enroll / varolan anonim ID'yi isimlendirme
- Face ID bulunamadı (404), geçersiz dosya (400), boş dosya (400)
- Yüz geçmişi ve işlem detayı sorgulama
- Loglama hatasının ana işlemi engellememesi

### LFW Doğruluk Testi

Labeled Faces in the Wild (LFW) veri kümesi üzerinde uçtan uca doğruluk testi yapar. Test sırasında API'nin (`docker compose up --build`) çalışıyor olması gerekir.

**LFW veri kümesini indirme:**
```bash
python3 -c "
from sklearn.datasets import fetch_lfw_people
fetch_lfw_people(data_home='~/scikit_learn_data')
"
```
Veri kümesi `~/scikit_learn_data/lfw_home/lfw_funneled/` yoluna indirilir. Her alt klasör bir kişidir, klasör adı kişinin ismidir.

**Testi çalıştırma:**
```bash
pip install requests scikit-learn pillow
python3 tests/e2e/test_lfw_accuracy.py
```

**Testin yaptıkları:**

1. `lfw_funneled/` içindeki tüm kişi klasörlerini tarar
2. En az `N` fotoğrafı olan kişileri seçer (`--min-photos N`, varsayılan: 2)
3. Her kişinin ilk fotoğrafını `POST /faces/recognize` ile tanıtır, dönen `faceId`'yi `POST /faces/enroll` ile kişinin klasör adını kullanarak isimlendirir
4. Kalan fotoğrafları test havuzuna ayırır
5. Test havuzundan rastgele `K` fotoğraf seçer (`--test-count K`, varsayılan: 20)
6. Her test fotoğrafını `POST /faces/recognize` ile tanır
7. Dönen `name` alanı ile gerçek kişi adını karşılaştırır
8. Rapor basar: toplam test, doğru tanıma, yanlış tanıma (başkasını dedi), tanıyamama (`new_anonymous` döndü), doğruluk yüzdesi
9. Yanlış eşleşmeleri listeler

**Parametreler:**
```bash
# En az 3 fotoğrafı olan kişiler, 50 test fotoğrafı
python3 tests/e2e/test_lfw_accuracy.py --min-photos 3 --test-count 50
```

## Proje Yapısı

```
face-recognition-api/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI uygulaması, startup event (pgvector extension + tablo oluşturma)
│   ├── config.py            # Ortam değişkenleri (DATABASE_URL, SIMILARITY_THRESHOLD, PORT, INSIGHTFACE_MODEL_NAME)
│   ├── database.py          # SQLAlchemy engine, session, Base
│   ├── models.py            # Identity, FaceEmbedding, Process, ProcessFace tabloları
│   ├── schemas.py           # Pydantic request/response modelleri
│   ├── face_engine.py       # InsightFace sarmalayıcı (yüz tespiti + embedding çıkarma)
│   ├── recognition.py       # Tanıma mantığı: vektör arama, threshold karşılaştırma, enroll, loglama, yüz kırpma
│   └── routers/
│       ├── __init__.py
│       ├── faces.py         # /faces/* endpoint'leri
│       └── processes.py     # /processes/* endpoint'leri
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # pytest fixture'lar (mock FaceEngine, mock DB)
│   ├── test_recognition.py  # Birim testleri (15 test)
│   ├── test_routers.py      # API testleri (14 test)
│   └── e2e/
│       └── test_lfw_accuracy.py  # LFW ile doğruluk testi
├── fe/
│   └── index.html           # Web arayüzü (HTML+CSS+JS, / adresinde sunulur)
├── Dockerfile               # python:3.12-slim tabanlı, model imaja gömülü
├── docker-compose.yml       # PostgreSQL + pgvector + API
├── .dockerignore
├── requirements.txt
├── download_models.py       # Build sırasında InsightFace modelini indirir
├── .env                     # (opsiyonel) Ortam değişkenleri
├── PLAN.md                  # Mimari tasarım dokümanı
└── README.md
```

## Notlar

- **Lisans:** InsightFace (`buffalo_l` modeli) ticari olmayan kullanım için ücretsizdir. Ticari kullanım için InsightFace lisans koşullarını inceleyin.
- **Performans:** Yüz tanıma CPU üzerinde (`CPUExecutionProvider`) çalışır. Tek bir yüz için ~100-200ms, beş yüz için ~300-500ms civarındadır. GPU gerekmez.
- **Veritabanı arama:** Vektör aramaları PostgreSQL `<=>` (kosinüs mesafesi) operatörü ile yapılır. Veritabanı büyüdükçe performans için `face_embeddings` tablosuna HNSW indeksi eklenebilir (`USING hnsw (embedding vector_cosine_ops)`).
