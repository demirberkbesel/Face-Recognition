"""
Kalite kontrol birim testleri:
- Bulanıklık tespiti (check_blur)
- Düşük detection confidence → qualityWarning
- Bulanık görsel reddi
"""

import uuid
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from app.quality import check_blur
from app.quality import check_blur as real_check_blur
from app.recognition import recognize_faces, enroll_face
from app.config import MIN_DETECTION_CONFIDENCE, BLUR_THRESHOLD
from tests.conftest import MOCK_BBOX, MOCK_EMBEDDING, MOCK_FACE_ID


class TestBlurDetection:
    def test_sharp_image_returns_not_blurry(self):
        """Net bir görüntü bulanık sayılmamalı."""
        sharp = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        is_blurry, variance = check_blur(sharp, BLUR_THRESHOLD)
        assert not is_blurry
        assert variance >= BLUR_THRESHOLD

    def test_blurry_image_returns_blurry(self):
        """Düz renk (bulanık) görüntü bulanık sayılmalı."""
        blurry = np.ones((200, 200, 3), dtype=np.uint8) * 128
        is_blurry, variance = check_blur(blurry, BLUR_THRESHOLD)
        assert is_blurry
        assert variance < BLUR_THRESHOLD

    def test_zero_threshold_never_blurry(self):
        """Eşik 0 ise hiçbir görüntü bulanık sayılmamalı."""
        flat = np.ones((100, 100, 3), dtype=np.uint8) * 128
        is_blurry, _ = check_blur(flat, threshold=0.0)
        assert not is_blurry

    def test_high_threshold_all_blurry(self):
        """Çok yüksek eşikte tüm görüntüler bulanık sayılmalı."""
        sharp = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        is_blurry, variance = check_blur(sharp, threshold=99999.0)
        assert is_blurry


class TestBlurInRecognize:
    def test_blurry_image_raises_error(self, mock_db):
        """Bulanık görsel recognize'da ValueError fırlatmalı."""
        flat = np.ones((100, 100, 3), dtype=np.uint8) * 128
        with patch("app.recognition.check_blur", wraps=real_check_blur):
            with patch("app.recognition.BLUR_THRESHOLD", 1000.0):
                with pytest.raises(ValueError, match="bulanık"):
                    recognize_faces(flat, mock_db)

    def test_sharp_image_passes(self, mock_empty_face_engine, mock_db):
        """Net görsel normal akışta çalışmalı (blur hatası vermemeli)."""
        sharp = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        with patch("app.recognition.check_blur", wraps=real_check_blur):
            result = recognize_faces(sharp, mock_db)
        assert result["faceCount"] == 0  # mock engine hiç yüz döndürmedi


class TestLowConfidence:
    def test_low_detection_confidence_adds_warning(self, mock_db):
        """Düşük det_score → qualityWarning dönmeli."""
        with patch("app.recognition.engine") as mock_engine:
            mock_engine.detect_faces.return_value = [
                {
                    "bbox": MOCK_BBOX,
                    "embedding": MOCK_EMBEDDING,
                    "det_score": 0.3,  # < MIN_DETECTION_CONFIDENCE
                }
            ]
            mock_db.execute.return_value.fetchone.return_value = None

            with patch("app.recognition.MIN_DETECTION_CONFIDENCE", 0.5):
                result = recognize_faces(
                    np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8),
                    mock_db,
                )

        assert result["faceCount"] == 1
        assert result["faces"][0]["qualityWarning"] is not None
        assert "düşük" in result["faces"][0]["qualityWarning"].lower()

    def test_high_detection_confidence_no_warning(self, mock_db):
        """Yüksek det_score → qualityWarning null olmalı."""
        with patch("app.recognition.engine") as mock_engine:
            mock_engine.detect_faces.return_value = [
                {
                    "bbox": MOCK_BBOX,
                    "embedding": MOCK_EMBEDDING,
                    "det_score": 0.9,  # >= MIN_DETECTION_CONFIDENCE
                }
            ]
            mock_db.execute.return_value.fetchone.return_value = None

            with patch("app.recognition.MIN_DETECTION_CONFIDENCE", 0.5):
                result = recognize_faces(
                    np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8),
                    mock_db,
                )

        assert result["faceCount"] == 1
        assert result["faces"][0]["qualityWarning"] is None


class TestBlurInEnroll:
    def test_blurry_enroll_raises_error(self, mock_db):
        """Bulanık görsel ile enroll ValueError fırlatmalı."""
        flat = np.ones((100, 100, 3), dtype=np.uint8) * 128
        with patch("app.recognition.check_blur", wraps=real_check_blur):
            with patch("app.recognition.BLUR_THRESHOLD", 1000.0):
                with pytest.raises(ValueError, match="bulanık"):
                    enroll_face(flat, "Test", None, mock_db)
