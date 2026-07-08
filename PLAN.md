# Face Recognition API - Sistem Mimarisi ve Tasarım Planı (PLAN.md)

Bu belge, **Face Recognition API** projesinin tüm gereksinimlerini, teknik kararlarını, veritabanı şemasını, API kontratlarını ve Dockerizasyon stratejisini detaylandıran üst düzey tasarım ve planlama kılavuzudur.

---

## 1. Genel Mimari Bakış

Sistem, asenkron ve yüksek performanslı bir API katmanı, derin öğrenme tabanlı yüz tespiti/tanıma modülü ve vektör aramaları için optimize edilmiş bir ilişkisel veritabanından oluşur.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           İstemci (Client)                              │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ HTTP (UploadFile / JSON)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      FastAPI Uygulama Katmanı                           │
│                                                                         │
│  ┌───────────────────────┐ ┌──────────────────────┐ ┌────────────────┐  │
│  │   API Endpoints       │ │  InsightFace Engine  │ │  Background    │  │
│  │   (FastAPI/Pydantic)  │ │ (Yüz Tespit/Tanıma)  │ │  Log Yazıcı    │  │
│  └───────────┬───────────┘ └──────────┬───────────┘ └───────┬────────┘  │
└──────────────┼────────────────────────┼─────────────────────┼───────────┘
               │                        │                     │
               │ SQL Queries / Vector Search (Cosine Dist)     │ Async Logging
               ▼                        ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 PostgreSQL Veritabanı (+ pgvector)                      │
│                                                                         │
│  ┌───────────────────────┐ ┌──────────────────────┐ ┌────────────────┐  │
│  │      identities       │ │   face_embeddings    │ │   processes    │  │
│  │   (Face ID / Name)    │ │   (512-dim Vector)   │ │  (Process ID)  │  │
│  └───────────────────────┘ └──────────────────────┘ └────────────────┘  │
│                                                     ┌────────────────┐  │
│                                                     │ process_faces  │  │
│                                                     │ (Log Detayları)│  │
│                                                     └────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Ana Bileşenler:
1. **API Katmanı (FastAPI):** Python tabanlı asenkron web çatısı. Pydantic ile girdi-çıktı doğrulama ve OpenAPI dokümantasyonu otomatik olarak sağlanacaktır.
2. **Yüz Tanıma Katmanı (InsightFace):** Yüz tespiti ve 512 boyutlu yüz öznitelik vektörlerinin (embeddings) çıkarılması için endüstri standardı olan ONNX Runtime destekli InsightFace kullanılacaktır.
3. **Veri Katmanı (PostgreSQL + pgvector):** İlişkisel verilerin saklanması ve yüz vektörlerinin kosinüs mesafesiyle (cosine distance) veritabanı seviyesinde hızlıca aranması için `pgvector` eklentisi kullanılacaktır.

---

## 2. Veritabanı Şeması Tasarımı (PostgreSQL + pgvector)

PostgreSQL üzerinde `pgvector` eklentisi aktif edilerek (create extension vector) aşağıdaki tablolar kurulacaktır.

### 2.1. Tablo İlişkileri ve Alanlar

#### 1. `identities` Tablosu
Kişilerin (tanınan veya anonim) ana kayıtlarını tutar.
```sql
CREATE TABLE identities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status VARCHAR(20) NOT NULL, -- 'known', 'anonymous'
    name VARCHAR(255) NULL,       -- Sadece 'known' ise dolu, aksi halde NULL
    metadata JSONB NULL,          -- Ekstra metaveriler
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_identities_status ON identities(status);
```

#### 2. `face_embeddings` Tablosu
Bir kişiye ait 512 boyutlu yüz vektörlerini tutar. Birden fazla vektör aynı kişiye bağlanabilir.
```sql
CREATE TABLE face_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    identity_id UUID NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
    embedding vector(512) NOT NULL, -- pgvector eklentisinin tipi
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
-- Hızlı kosinüs mesafesi aramaları için HNSW indeksi kurulur
CREATE INDEX idx_face_embeddings_cosine ON face_embeddings USING hnsw (embedding vector_cosine_ops);
```

#### 3. `processes` Tablosu
API çağrılarını ve işlem bazlı ana logları tutar.
```sql
CREATE TABLE processes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    task_details JSONB NOT NULL -- İşlem tipi, toplam yüz sayısı vb. genel özet
);
```

#### 4. `process_faces` Tablosu (Ara Tablo / Detaylı Log)
Bir işlemin hangi yüzlerle, hangi bounding box'larla ve ne kadarlık bir benzerlik skoruyla sonuçlandığını tutan ilişkisel geçmiş tablosudur.
```sql
CREATE TABLE process_faces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    process_id UUID NOT NULL REFERENCES processes(id) ON DELETE CASCADE,
    identity_id UUID NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
    status VARCHAR(20) NOT NULL,       -- İşlem anındaki durum ('known', 'anonymous', 'new_anonymous')
    confidence FLOAT NOT NULL,         -- Eşleşme doğruluk skoru (0.0 - 1.0)
    bounding_box JSONB NOT NULL,       -- [xmin, ymin, xmax, ymax] koordinatları
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_process_faces_identity ON process_faces(identity_id);
CREATE INDEX idx_process_faces_process ON process_faces(process_id);
```

---

## 3. Algoritma ve Karar Mantığı

Bir yüz tanıma (`/faces/recognize`) isteği geldiğinde aşağıdaki süreç işletilecektir:

1. **Görüntü Okuma ve Doğrulama:**
   * Gelen dosyanın formatı doğrulanır (JPEG, PNG vb.).
   * Görsel OpenCV veya PIL ile belleğe yüklenir. Bozuk veya okunamayan görüntüler için HTTP 400 döner.

2. **Yüz Tespiti (Face Detection):**
   * InsightFace dedektörü ile görseldeki tüm yüzler ve koordinatları (`bounding_box`) tespit edilir.
   * Yüz bulunamazsa hata dönülmez; boş yüz listesi ve `process_id` içeren HTTP 200 başarılı yanıtı döner.

3. **Öznitelik Çıkarma (Embedding Extraction):**
   * Tespit edilen her yüz için 512 boyutlu `vector` hesaplanır.

4. **Vektör Arama ve Tanıma (Recognition):**
   * Çıkartılan her vektör için PostgreSQL'de şu SQL sorgusu çalıştırılır (Kosinüs Mesafesi araması):
     ```sql
     SELECT identity_id, (embedding <=> :current_embedding) as distance
     FROM face_embeddings
     ORDER BY distance ASC
     LIMIT 1;
     ```
     *(Not: `<=>` operatörü pgvector'de kosinüs mesafesini temsil eder. Mesafe ne kadar küçükse benzerlik o kadar yüksektir).*
   * **Benzerlik Skoru Hesaplama:** `confidence = 1.0 - distance` (veya model tipine göre normalize edilmiş benzerlik yüzdesi).
   * **Threshold (Eşik) Karşılaştırması:**
     * Eğer `distance < (1.0 - SIMILARITY_THRESHOLD)` ise (yani benzerlik eşiğin üzerindeyse):
       * Eşleşen `identity_id` alınır.
       * İlgili kimliğin `status` değeri veritabanından çekilir (`known` veya `anonymous`).
       * Bu işlemdeki durum: `known` veya `anonymous` olur.
     * Eğer `distance >= (1.0 - SIMILARITY_THRESHOLD)` ise (yani benzerlik eşiğin altındaysa):
       * Bu yüz yeni bir yüzdür.
       * `identities` tablosunda `status = 'anonymous'` olan yeni bir UUID (`faceId`) kaydı oluşturulur.
       * `face_embeddings` tablosuna bu yeni `identity_id` ile yüz vektörü eklenir.
       * Bu işlemdeki durum: `new_anonymous` olur.

5. **Loglama (Hata Toleranslı ve Asenkron):**
   * Yüz tanıma işlemi tamamlandıktan sonra, FastAPI `BackgroundTasks` mekanizması kullanılarak arka planda `processes` ve `process_faces` tablolarına kayıt atılır.
   * Loglama esnasında oluşabilecek veritabanı hataları yakalanır (loglanır) ancak istemciye dönecek olan HTTP cevabını engellemez.

---

## 4. API Endpoints ve Sözleşmeleri (Contracts)

Tüm istekler ve cevaplar standart JSON formatında olacaktır.

### 4.1. POST /faces/recognize
Görüntüdeki tüm yüzleri tespit eder, tanımlar veya anonim olarak kaydeder.

*   **Request:**
    *   `Content-Type: multipart/form-data`
    *   `image` (File): Yüklenecek görsel dosya.
*   **Response (HTTP 200 - Yüz Bulundu):**
    ```json
    {
      "processId": "c9a6331a-6dfa-4be7-805c-3f98016fbe85",
      "faceCount": 2,
      "faces": [
        {
          "faceId": "8f309a9d-b8d2-4fe1-ba54-208b0a17409f",
          "status": "known",
          "name": "Ahmet Yılmaz",
          "metadata": {
            "title": "Software Engineer",
            "department": "R&D"
          },
          "boundingBox": {
            "xmin": 120,
            "ymin": 80,
            "xmax": 240,
            "ymax": 210
          },
          "confidence": 0.89
        },
        {
          "faceId": "47cb324c-1e24-4f9e-bd9d-bc11d290b0e5",
          "status": "new_anonymous",
          "name": null,
          "metadata": null,
          "boundingBox": {
            "xmin": 450,
            "ymin": 95,
            "xmax": 560,
            "ymax": 220
          },
          "confidence": 0.0
        }
      ]
    }
    ```
*   **Response (HTTP 200 - Yüz Bulunamadı):**
    ```json
    {
      "processId": "c9a6331a-6dfa-4be7-805c-3f98016fbe85",
      "faceCount": 0,
      "faces": []
    }
    ```

---

### 4.2. POST /faces/enroll
Yeni bir yüzü veya var olan bir anonim yüzü isimlendirerek kaydeder.

*   **Request (Yeni Kayıt - Image ile):**
    *   `Content-Type: multipart/form-data`
    *   `image` (File): Yüzün tespiti için görsel.
    *   `name` (String): Kişi ismi.
    *   `metadata` (Stringified JSON, Opsiyonel): Ekstra bilgiler.
*   **Request (Anonim Kaydı Güncelleme - Sadece JSON):**
    *   `Content-Type: application/json`
    ```json
    {
      "faceId": "47cb324c-1e24-4f9e-bd9d-bc11d290b0e5",
      "name": "Mehmet Demir",
      "metadata": {
        "role": "Designer"
      }
    }
    ```
*   **Response (HTTP 201 - Başarılı):**
    ```json
    {
      "faceId": "47cb324c-1e24-4f9e-bd9d-bc11d290b0e5",
      "status": "known",
      "name": "Mehmet Demir",
      "metadata": {
        "role": "Designer"
      }
    }
    ```

---

### 4.3. GET /faces/{faceId}
Bir face ID'nin detaylarını sorgular.

*   **Response (HTTP 200):**
    ```json
    {
      "faceId": "47cb324c-1e24-4f9e-bd9d-bc11d290b0e5",
      "status": "known",
      "name": "Mehmet Demir",
      "metadata": {
        "role": "Designer"
      }
    }
    ```
*   **Response (HTTP 404 - Bulunamadı):**
    ```json
    {
      "detail": "Face ID bulunamadı."
    }
    ```

---

### 4.4. DELETE /faces/{faceId}
Bir kimliği ve ona bağlı tüm yüz vektörlerini sistemden siler.

*   **Response (HTTP 200):**
    ```json
    {
      "message": "Face ID ve bağlı vektörler başarıyla silindi.",
      "faceId": "47cb324c-1e24-4f9e-bd9d-bc11d290b0e5"
    }
    ```

---

### 4.5. GET /faces/{faceId}/history
Bir face ID'nin daha önce hangi process'lerde, ne zaman, hangi koordinatlarda ve güven skoruyla yer aldığını döner.

*   **Response (HTTP 200):**
    ```json
    {
      "faceId": "47cb324c-1e24-4f9e-bd9d-bc11d290b0e5",
      "history": [
        {
          "processId": "c9a6331a-6dfa-4be7-805c-3f98016fbe85",
          "timestamp": "2026-07-06T12:30:45Z",
          "status": "new_anonymous",
          "confidence": 0.0,
          "boundingBox": {
            "xmin": 450,
            "ymin": 95,
            "xmax": 560,
            "ymax": 220
          }
        },
        {
          "processId": "e5b8332a-7dfa-4be7-805c-3f98016fa999",
          "timestamp": "2026-07-06T14:15:22Z",
          "status": "known",
          "confidence": 0.94,
          "boundingBox": {
            "xmin": 448,
            "ymin": 96,
            "xmax": 558,
            "ymax": 218
          }
        }
      ]
    }
    ```

---

### 4.6. GET /processes/{processId}
Bir işlemin (process) detaylarını ve sonucunu geri çağırır.

*   **Response (HTTP 200):**
    ```json
    {
      "processId": "c9a6331a-6dfa-4be7-805c-3f98016fbe85",
      "timestamp": "2026-07-06T12:30:45Z",
      "taskDetails": {
        "type": "recognition",
        "faceCount": 2
      },
      "faces": [
        {
          "faceId": "8f309a9d-b8d2-4fe1-ba54-208b0a17409f",
          "status": "known",
          "confidence": 0.89,
          "boundingBox": {
            "xmin": 120,
            "ymin": 80,
            "xmax": 240,
            "ymax": 210
          }
        },
        {
          "faceId": "47cb324c-1e24-4f9e-bd9d-bc11d290b0e5",
          "status": "new_anonymous",
          "confidence": 0.0,
          "boundingBox": {
            "xmin": 450,
            "ymin": 95,
            "xmax": 560,
            "ymax": 220
          }
        }
      ]
    }
    ```

---

## 5. Docker ve Canlandırma (Deployment) Stratejisi

Sistem `docker-compose` kullanılarak tek bir komutla ayağa kaldırılabilecek şekilde tasarlanacaktır.

### 5.1. Dockerfile Tasarımı (Offline Model Gömme)
InsightFace modelleri (`buffalo_l` veya `buffalo_m`) Docker build aşamasında önceden indirilecek ve imajın içerisine yerleştirilecektir. Bu işlem `dlib` ve `onnxruntime` bağımlılıklarının derlenme sürelerini optimize edecek şekilde çok aşamalı (multi-stage) veya önceden derlenmiş tekerlekler (pre-compiled wheels) kullanılarak yapılacaktır.

```dockerfile
# Örnek Dockerfile yapısı
FROM python:3.10-slim

# Gerekli sistem kütüphaneleri (OpenCV ve derleme araçları için)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model dosyalarının önceden indirilmesini sağlayan hazırlık betiği
COPY download_models.py .
RUN python download_models.py

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5.2. `docker-compose.yml` Konfigürasyonu
Hem PostgreSQL + `pgvector` hem de API servisimizi kalıcı veri yolları (volumes) ile yapılandıracağız.

```yaml
version: '3.8'

services:
  db:
    image: ankane/pgvector:v0.5.1 # pgvector önceden kurulu resmi PostgreSQL imajı
    container_name: face_recognition_db
    environment:
      POSTGRES_DB: face_db
      POSTGRES_USER: face_user
      POSTGRES_PASSWORD: face_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U face_user -d face_db"]
      interval: 5s
      timeout: 5s
      retries: 5

  api:
    build: .
    container_name: face_recognition_api
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://face_user:face_password@db:5432/face_db
      - SIMILARITY_THRESHOLD=0.45
      - INSIGHTFACE_MODEL_NAME=buffalo_l
      - PORT=8000
    depends_on:
      db:
        condition: service_healthy
    volumes:
      - .:/app

volumes:
  postgres_data:
```

---

## 6. Proje Yol Haritası ve Geliştirme Adımları

Planlama sonrası yapılacak geliştirme aşamaları aşağıdaki gibi olacaktır:

1.  **Aşama 1: Çevre Kurulumu:** Docker, PostgreSQL/pgvector konfigürasyonlarının yapılması ve veritabanı şemasının hazırlanması.
2.  **Aşama 2: Çekirdek ML Modülü:** InsightFace entegrasyonu, model indirme betiğinin hazırlanması ve yüz tespiti/öznitelik çıkarımı testleri.
3.  **Aşama 3: Veritabanı ve Arama Mantığı:** SQLAlchemy/SQLModel ile veritabanı bağlantısı, kosinüs mesafesi aramaları ve threshold eşleştirme fonksiyonlarının yazılması.
4.  **Aşama 4: API Geliştirme:** FastAPI uç noktalarının, girdi/çıktı şemalarının ve hata kontrol mekanizmalarının tamamlanması.
5.  **Aşama 5: Loglama ve Geçmiş:** Asenkron loglama ara tablolarının entegrasyonu ve geçmiş sorgulama endpoint'lerinin testi.
6.  **Aşama 6: Doğrulama ve Testler:** Unit/Integration testlerinin yazılması, performans ölçümleri ve Docker build doğrulaması.
