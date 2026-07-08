import uuid
from unittest.mock import MagicMock, patch
from datetime import datetime

import numpy as np
import pytest
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.recognition import (
    recognize_faces,
    enroll_face,
    enroll_existing_face,
    _find_nearest_match,
    _log_process_background,
)
from app.config import SIMILARITY_THRESHOLD
from tests.conftest import MOCK_BBOX, MOCK_EMBEDDING, MOCK_FACE_ID, make_mock_identity


class TestRecognizeFaces:
    def test_no_faces_detected(self, mock_empty_face_engine, mock_db):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = recognize_faces(image, mock_db)

        assert result["faceCount"] == 0
        assert result["faces"] == []
        assert isinstance(uuid.UUID(result["processId"]), uuid.UUID)
        mock_db.commit.assert_called_once()

    def test_new_anonymous_face(self, mock_face_engine, mock_db):
        mock_db.execute.return_value.fetchone.return_value = None

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = recognize_faces(image, mock_db)

        assert result["faceCount"] == 1
        assert uuid.UUID(result["faces"][0]["faceId"])
        assert result["faces"][0]["status"] == "new_anonymous"
        assert result["faces"][0]["name"] is None
        assert result["faces"][0]["boundingBox"] == MOCK_BBOX
        assert result["faces"][0]["confidence"] == 0.0
        mock_db.add.assert_called()
        mock_db.commit.assert_called_once()

    def test_known_face_match(self, mock_face_engine, mock_db):
        distance = 0.15
        mock_db.execute.return_value.fetchone.return_value = (MOCK_FACE_ID, distance)
        mock_identity = make_mock_identity(
            identity_id=MOCK_FACE_ID,
            status="known",
            name="Ahmet Yılmaz",
            extra_data={"department": "R&D"},
        )

        def query_side_effect(model):
            q = MagicMock()
            if model.__name__ == "Identity":
                q.filter.return_value.first.return_value = mock_identity
            return q

        mock_db.query.side_effect = query_side_effect

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = recognize_faces(image, mock_db)

        assert result["faceCount"] == 1
        assert result["faces"][0]["faceId"] == str(MOCK_FACE_ID)
        assert result["faces"][0]["status"] == "known"
        assert result["faces"][0]["name"] == "Ahmet Yılmaz"
        assert result["faces"][0]["metadata"] == {"department": "R&D"}
        assert result["faces"][0]["confidence"] == 0.85
        mock_db.commit.assert_called_once()

    def test_anonymous_face_match(self, mock_face_engine, mock_db):
        distance = 0.20
        anon_id = uuid.uuid4()
        mock_db.execute.return_value.fetchone.return_value = (anon_id, distance)
        mock_identity = make_mock_identity(
            identity_id=anon_id,
            status="anonymous",
            name=None,
            extra_data=None,
        )

        def query_side_effect(model):
            q = MagicMock()
            if model.__name__ == "Identity":
                q.filter.return_value.first.return_value = mock_identity
            return q

        mock_db.query.side_effect = query_side_effect

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = recognize_faces(image, mock_db)

        assert result["faceCount"] == 1
        assert result["faces"][0]["faceId"] == str(anon_id)
        assert result["faces"][0]["status"] == "anonymous"
        assert result["faces"][0]["name"] is None
        assert result["faces"][0]["metadata"] is None
        assert result["faces"][0]["confidence"] == 0.80

    def test_below_threshold_becomes_new_anonymous(self, mock_face_engine, mock_db):
        large_distance = 0.70
        mock_db.execute.return_value.fetchone.return_value = (MOCK_FACE_ID, large_distance)

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = recognize_faces(image, mock_db)

        assert result["faceCount"] == 1
        assert uuid.UUID(result["faces"][0]["faceId"])
        assert result["faces"][0]["status"] == "new_anonymous"
        assert result["faces"][0]["confidence"] == 0.0

    def test_multiple_faces_same_image(self, mock_db):
        with patch("app.recognition.engine") as mock_engine:
            mock_engine.detect_faces.return_value = [
                {"bbox": MOCK_BBOX, "embedding": MOCK_EMBEDDING, "det_score": 0.95},
                {"bbox": MOCK_BBOX, "embedding": MOCK_EMBEDDING, "det_score": 0.90},
            ]

            def execute_side_effect(*args, **kwargs):
                if "LIMIT 1" in str(args[0]):
                    return MagicMock(fetchone=lambda: None)
                return MagicMock()

            mock_db.execute.side_effect = execute_side_effect

            image = np.zeros((100, 100, 3), dtype=np.uint8)
            result = recognize_faces(image, mock_db)

            assert result["faceCount"] == 2
            assert uuid.UUID(result["faces"][0]["faceId"])
            assert uuid.UUID(result["faces"][1]["faceId"])
            assert result["faces"][0]["faceId"] != result["faces"][1]["faceId"]
            mock_db.commit.assert_called_once()

    def test_background_log_called(self, mock_face_engine, mock_db, mock_session_local):
        mock_db.execute.return_value.fetchone.return_value = None

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = recognize_faces(image, mock_db)

        mock_session_local_instance = mock_session_local.return_value
        mock_session_local_instance.add.assert_called()
        mock_session_local_instance.commit.assert_called_once()


class TestEnroll:
    def test_enroll_face_success(self, mock_face_engine, mock_db):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = enroll_face(image, "Mehmet Demir", {"role": "Developer"}, mock_db)

        assert uuid.UUID(result["faceId"])
        assert result["status"] == "known"
        assert result["name"] == "Mehmet Demir"
        assert result["metadata"] == {"role": "Developer"}
        mock_db.add.assert_called()

    def test_enroll_face_no_face(self, mock_empty_face_engine, mock_db):
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Görselde yüz bulunamadı"):
            enroll_face(image, "Mehmet Demir", None, mock_db)

    def test_enroll_existing_face(self, mock_db):
        identity = make_mock_identity(
            identity_id=MOCK_FACE_ID,
            status="anonymous",
            name=None,
        )
        mock_db.query.return_value.filter.return_value.first.return_value = identity

        result = enroll_existing_face(MOCK_FACE_ID, "Ayşe Kaya", {"title": "Designer"}, mock_db)

        assert result["faceId"] == str(MOCK_FACE_ID)
        assert result["status"] == "known"
        assert result["name"] == "Ayşe Kaya"
        assert result["metadata"] == {"title": "Designer"}
        assert identity.status == "known"
        assert identity.name == "Ayşe Kaya"
        assert identity.extra_data == {"title": "Designer"}

    def test_enroll_existing_face_not_found(self, mock_db):
        mock_db.query.return_value.filter.return_value.first.return_value = None

        with pytest.raises(ValueError, match="Face ID bulunamadı"):
            enroll_existing_face(uuid.uuid4(), "Test", None, mock_db)


class TestFindNearestMatch:
    def test_find_match_found(self, mock_db):
        expected_id = uuid.uuid4()
        expected_distance = 0.25
        mock_db.execute.return_value.fetchone.return_value = (expected_id, expected_distance)

        result = _find_nearest_match(MOCK_EMBEDDING, mock_db)

        assert result is not None
        assert result["identity_id"] == expected_id
        assert result["distance"] == expected_distance

    def test_find_match_not_found(self, mock_db):
        mock_db.execute.return_value.fetchone.return_value = None

        result = _find_nearest_match(MOCK_EMBEDDING, mock_db)
        assert result is None


class TestLogProcessBackground:
    def test_log_success(self, mock_session_local):
        mock_instance = mock_session_local.return_value
        process_id = uuid.uuid4()
        detected_faces = [{"bbox": MOCK_BBOX}]
        identity_entries = [
            {"identity_id": uuid.uuid4(), "status": "new_anonymous", "confidence": 0.0, "bbox": MOCK_BBOX}
        ]

        _log_process_background(process_id, detected_faces, identity_entries)

        mock_instance.add.assert_called()
        mock_instance.commit.assert_called_once()
        mock_instance.close.assert_called_once()

    def test_log_failure_does_not_raise(self, mock_session_local):
        mock_instance = mock_session_local.return_value
        mock_instance.commit.side_effect = Exception("DB error")

        _log_process_background(uuid.uuid4(), [], [])
