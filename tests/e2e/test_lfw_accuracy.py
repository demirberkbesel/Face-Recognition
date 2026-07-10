#!/usr/bin/env python3
"""
LFW dataset ile E2E doğruluk testi.
Kullanım:  python tests/e2e/test_lfw_accuracy.py [--min-photos N] [--test-count K]

Önce docker compose up --build ile API'yi ayağa kaldırın,
sonra bu script'i çalıştırın.
"""

import os
import sys
import random
import argparse
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Ayarlar
# ---------------------------------------------------------------------------
API_BASE = "http://127.0.0.1:8000"
LFW_PATH = os.path.expanduser("~/scikit_learn_data/lfw_home/lfw_funneled")

# ---------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ---------------------------------------------------------------------------

def get_person_photos(person_dir: str, min_photos: int = 2) -> Optional[list[str]]:
    """
    Bir kişi klasöründeki tüm .jpg dosyalarını döndürür.
    Eğer en az min_photos kadar fotoğraf yoksa None döndürür.
    """
    photos = sorted(
        [os.path.join(person_dir, f) for f in os.listdir(person_dir) if f.lower().endswith(".jpg")]
    )
    return photos if len(photos) >= min_photos else None


def recognize(photo_path: str) -> Optional[dict]:
    """
    POST /faces/recognize → fotoğraftaki yüzleri tespit et.
    Başarılı olursa ilk yüzün bilgilerini döndürür, yoksa None.
    """
    with open(photo_path, "rb") as f:
        resp = requests.post(
            f"{API_BASE}/faces/recognize",
            files={"file": (os.path.basename(photo_path), f, "image/jpeg")},
            timeout=30,
        )
    if resp.status_code != 200:
        print(f"  [UYARI] /recognize {resp.status_code}: {resp.text[:100]}")
        return None

    data = resp.json()
    if data["faceCount"] == 0:
        return None
    return data["faces"][0]


def enroll(face_id: str, name: str) -> bool:
    """
    POST /faces/enroll → anonim bir yüz kimliğini isimlendir.
    """
    resp = requests.post(
        f"{API_BASE}/faces/enroll",
        data={"face_id": face_id, "name": name},
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"  [UYARI] /enroll {resp.status_code}: {resp.text[:100]}")
        return False
    return True


# ---------------------------------------------------------------------------
# E2E test
# ---------------------------------------------------------------------------

def run_e2e_test(min_photos: int = 2, test_count: int = 20):
    print("=" * 60)
    print("  LFW Doğruluk Testi")
    print("=" * 60)
    print(f"  Dataset      : {LFW_PATH}")
    print(f"  Min fotoğraf : {min_photos}")
    print(f"  Test sayısı  : {test_count}")
    print(f"  API          : {API_BASE}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Dataset'i tara
    # -----------------------------------------------------------------------
    print("\n[1/6] Dataset taranıyor...")
    person_dirs = sorted([
        os.path.join(LFW_PATH, d)
        for d in os.listdir(LFW_PATH)
        if os.path.isdir(os.path.join(LFW_PATH, d))
    ])
    print(f"       Toplam kişi: {len(person_dirs)}")

    # -----------------------------------------------------------------------
    # 2. En az N fotoğrafı olan kişileri seç
    # -----------------------------------------------------------------------
    print(f"\n[2/6] En az {min_photos} fotoğrafı olan kişiler seçiliyor...")
    qualified = []   # (person_name, [photo_path, ...])
    for d in person_dirs:
        photos = get_person_photos(d, min_photos)
        if photos:
            person_name = os.path.basename(d)
            qualified.append((person_name, photos))

    print(f"       Seçilen kişi: {len(qualified)}")

    if not qualified:
        print("\n[HATA] Hiç uygun kişi bulunamadı. Dataset yolunu kontrol edin.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 3. Her kişi için: ilk fotoğrafı tanı, sonra isimlendir
    # -----------------------------------------------------------------------
    print("\n[3/6] Kayıt (enrollment) yapılıyor...")
    enrolled_count = 0
    test_pool = []   # (person_name, photo_path)

    for idx, (person_name, photos) in enumerate(qualified, 1):
        first_photo = photos[0]

        # 3a. Tanıma isteği → faceId al
        result = recognize(first_photo)
        if result is None:
            print(f"       [{idx}/{len(qualified)}] {person_name}: yüz tespit edilemedi, atlanıyor.")
            continue

        face_id = result["faceId"]

        # 3b. Enroll ile isimlendir
        if not enroll(face_id, person_name):
            print(f"       [{idx}/{len(qualified)}] {person_name}: enroll başarısız, atlanıyor.")
            continue

        enrolled_count += 1

        # 3c. Kalan fotoğrafları test havuzuna ekle
        for photo in photos[1:]:
            test_pool.append((person_name, photo))

        # İlerleme
        if idx % 50 == 0 or idx == len(qualified):
            print(f"       [{idx}/{len(qualified)}] {person_name} → kayıtlı")

    print(f"\n       Toplam kayıt     : {enrolled_count}")
    print(f"       Test havuzu      : {len(test_pool)} fotoğraf")

    if not test_pool:
        print("\n[HATA] Test havuzu boş. Daha fazla fotoğraflı kişi gerekli.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 4. Test havuzundan rastgele K fotoğraf seç
    # -----------------------------------------------------------------------
    print(f"\n[4/6] Test havuzundan {test_count} fotoğraf seçiliyor...")
    sample_size = min(test_count, len(test_pool))
    test_samples = random.sample(test_pool, sample_size)
    print(f"       Seçilen: {sample_size} fotoğraf")

    # -----------------------------------------------------------------------
    # 5. Her test fotoğrafını tanı
    # -----------------------------------------------------------------------
    print(f"\n[5/6] Test fotoğrafları tanınıyor...")

    results = []  # (expected_name, actual_name, actual_status, photo_path)

    for idx, (expected_name, photo_path) in enumerate(test_samples, 1):
        result = recognize(photo_path)

        if result is None:
            actual_name = None
            actual_status = "no_face"
        else:
            actual_name = result.get("name")
            actual_status = result.get("status")

        results.append((expected_name, actual_name, actual_status, photo_path))

        # İlerleme çubuğu
        bar_len = 30
        filled = int(bar_len * idx / len(test_samples))
        bar = "█" * filled + "░" * (bar_len - filled)
        pct = int(100 * idx / len(test_samples))
        print(f"       |{bar}| {idx}/{len(test_samples)} ({pct}%)", end="\r")

    print()

    # -----------------------------------------------------------------------
    # 6. Sonuçları raporla
    # -----------------------------------------------------------------------
    print(f"\n[6/6] Rapor hazırlanıyor...\n")
    print("=" * 60)
    print("  DOĞRULUK RAPORU")
    print("=" * 60)

    total = len(results)
    correct = 0          # Doğru kişi
    wrong = 0            # Başka bir kişi
    not_recognized = 0   # Yeni anonim / hiç yüz bulunamadı
    mismatches = []      # (expected, actual) detayları

    for expected, actual_name, actual_status, _ in results:
        if actual_status == "known" and actual_name == expected:
            correct += 1
        elif actual_status in ("new_anonymous", "anonymous", "no_face"):
            not_recognized += 1
        else:
            wrong += 1
            mismatches.append((expected, actual_name))

    accuracy = (correct / total) * 100 if total > 0 else 0.0

    print(f"  Toplam test         : {total}")
    print(f"  Doğru tanıma        : {correct}")
    print(f"  Yanlış tanıma       : {wrong}")
    print(f"  Tanıyamama          : {not_recognized}")
    print(f"  Doğruluk           : %{accuracy:.2f}")
    print()

    if mismatches:
        print("  Yanlış eşleşmeler:")
        print(f"  {'Gerçek kişi':<25} {'Tahmin':<25}")
        print(f"  {'-'*25} {'-'*25}")
        for expected, actual in mismatches[:20]:  # İlk 20 hata
            print(f"  {expected:<25} {actual or '<None>':<25}")
        if len(mismatches) > 20:
            print(f"  ... ve {len(mismatches) - 20} hata daha")
        print()

    print("=" * 60)
    return accuracy


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LFW ile E2E doğruluk testi")
    parser.add_argument(
        "--min-photos", type=int, default=2,
        help="Bir kişide aranacak en az fotoğraf sayısı (varsayılan: 2)",
    )
    parser.add_argument(
        "--test-count", type=int, default=20,
        help="Test havuzundan seçilecek rastgele fotoğraf sayısı (varsayılan: 20)",
    )
    args = parser.parse_args()

    run_e2e_test(min_photos=args.min_photos, test_count=args.test_count)
