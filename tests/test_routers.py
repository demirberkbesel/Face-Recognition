import uuid
import json
from unittest.mock import MagicMock, patch
from io import BytesIO

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.database import get_db
from app.models import Identity, Process, ProcessFace
from tests.conftest import MOCK_BBOX, MOCK_FACE_ID


def _valid_jpeg():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


VALID_JPEG = _valid_jpeg()


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.flush = MagicMock()
    db.commit = MagicMock()
    db.rollback = MagicMock()
    db.execute = MagicMock()
    db.query = MagicMock()
    db.add = MagicMock()
    return db


@pytest.fixture
def client(mock_db):
    app.dependency_overrides[get_db] = lambda: mock_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


MOCK_RECOGNIZE_RESPONSE = {
    "processId": str(uuid.uuid4()),
    "faceCount": 1,
    "faces": [
        {
            "faceId": str(MOCK_FACE_ID),
            "status": "known",
            "name": "Ahmet Yılmaz",
            "metadata": {"department": "R&D"},
            "boundingBox": MOCK_BBOX,
            "confidence": 0.89,
        }
    ],
}


class TestFacesEndpoints:
    def test_recognize_success(self, client, mock_db):
        with patch("app.routers.faces.recognize_faces") as mock_recognize:
            mock_recognize.return_value = MOCK_RECOGNIZE_RESPONSE
            response = client.post(
                "/faces/recognize",
                files={"file": ("test.jpg", VALID_JPEG, "image/jpeg")},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["faceCount"] == 1
        assert data["faces"][0]["faceId"] == str(MOCK_FACE_ID)
        assert data["faces"][0]["status"] == "known"

    def test_recognize_invalid_format(self, client, mock_db):
        response = client.post(
            "/faces/recognize",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400
        assert "Desteklenmeyen dosya formatı" in response.json()["detail"]

    def test_recognize_empty_file(self, client, mock_db):
        response = client.post(
            "/faces/recognize",
            files={"file": ("test.jpg", b"", "image/jpeg")},
        )
        assert response.status_code == 400
        assert "Boş dosya" in response.json()["detail"]

    def test_recognize_no_faces(self, client, mock_db):
        with patch("app.routers.faces.recognize_faces") as mock_recognize:
            mock_recognize.return_value = {
                "processId": str(uuid.uuid4()),
                "faceCount": 0,
                "faces": [],
            }
            response = client.post(
                "/faces/recognize",
                files={"file": ("test.jpg", VALID_JPEG, "image/jpeg")},
            )
        assert response.status_code == 200
        assert response.json()["faceCount"] == 0

    def test_enroll_with_image(self, client, mock_db):
        with patch("app.routers.faces.enroll_face") as mock_enroll:
            mock_enroll.return_value = {
                "faceId": str(MOCK_FACE_ID),
                "status": "known",
                "name": "Mehmet Demir",
                "metadata": {"role": "Developer"},
            }
            response = client.post(
                "/faces/enroll",
                data={"name": "Mehmet Demir", "metadata": json.dumps({"role": "Developer"})},
                files={"file": ("test.jpg", VALID_JPEG, "image/jpeg")},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Mehmet Demir"
        assert data["metadata"] == {"role": "Developer"}

    def test_enroll_existing_face(self, client, mock_db):
        with patch("app.routers.faces.enroll_existing_face") as mock_enroll:
            mock_enroll.return_value = {
                "faceId": str(MOCK_FACE_ID),
                "status": "known",
                "name": "Ayşe Kaya",
                "metadata": None,
            }
            response = client.post(
                "/faces/enroll",
                data={
                    "face_id": str(MOCK_FACE_ID),
                    "name": "Ayşe Kaya",
                },
            )
        assert response.status_code == 200
        assert response.json()["name"] == "Ayşe Kaya"

    def test_get_face_detail_found(self, client, mock_db):
        identity = MagicMock()
        identity.id = MOCK_FACE_ID
        identity.status = "known"
        identity.name = "Ahmet Yılmaz"
        identity.extra_data = {"department": "R&D"}
        mock_db.query.return_value.filter.return_value.first.return_value = identity

        response = client.get(f"/faces/{MOCK_FACE_ID}")
        assert response.status_code == 200
        data = response.json()
        assert data["faceId"] == str(MOCK_FACE_ID)
        assert data["status"] == "known"
        assert data["name"] == "Ahmet Yılmaz"

    def test_get_face_detail_not_found(self, client, mock_db):
        mock_db.query.return_value.filter.return_value.first.return_value = None

        response = client.get(f"/faces/{uuid.uuid4()}")
        assert response.status_code == 404

    def test_delete_face_success(self, client, mock_db):
        identity = MagicMock()
        identity.id = MOCK_FACE_ID
        mock_db.query.return_value.filter.return_value.first.return_value = identity

        response = client.delete(f"/faces/{MOCK_FACE_ID}")
        assert response.status_code == 200
        assert response.json()["faceId"] == str(MOCK_FACE_ID)

    def test_delete_face_not_found(self, client, mock_db):
        mock_db.query.return_value.filter.return_value.first.return_value = None

        response = client.delete(f"/faces/{uuid.uuid4()}")
        assert response.status_code == 404

    def test_get_face_history(self, client, mock_db):
        identity = MagicMock()
        identity.id = MOCK_FACE_ID
        mock_db.query.return_value.filter.return_value.first.return_value = identity

        pf = MagicMock()
        pf.process_id = uuid.uuid4()
        pf.identity_id = MOCK_FACE_ID
        pf.status = "known"
        pf.confidence = 0.89
        pf.bounding_box = MOCK_BBOX
        pf.created_at = MagicMock()

        process = MagicMock()
        process.id = pf.process_id
        process.timestamp = "2026-07-06T12:00:00Z"

        def query_side_effect(model):
            q = MagicMock()
            if model.__name__ == "ProcessFace":
                q.filter.return_value.order_by.return_value.all.return_value = [pf]
            elif model.__name__ == "Process":
                q.filter.return_value.first.return_value = process
            return q

        mock_db.query.side_effect = query_side_effect

        response = client.get(f"/faces/{MOCK_FACE_ID}/history")
        assert response.status_code == 200
        data = response.json()
        assert data["faceId"] == str(MOCK_FACE_ID)
        assert len(data["history"]) == 1
        assert data["history"][0]["status"] == "known"
        assert data["history"][0]["confidence"] == 0.89

    def test_get_face_history_not_found(self, client, mock_db):
        mock_db.query.return_value.filter.return_value.first.return_value = None

        response = client.get(f"/faces/{uuid.uuid4()}/history")
        assert response.status_code == 404


class TestProcessesEndpoints:
    def test_get_process_detail(self, client, mock_db):
        process = MagicMock()
        process.id = uuid.uuid4()
        process.timestamp = "2026-07-06T12:00:00Z"
        process.task_details = {"type": "recognition", "faceCount": 1}

        pf = MagicMock()
        pf.process_id = process.id
        pf.identity_id = MOCK_FACE_ID
        pf.status = "known"
        pf.confidence = 0.89
        pf.bounding_box = MOCK_BBOX

        identity = MagicMock()
        identity.id = MOCK_FACE_ID

        def query_side_effect(model):
            q = MagicMock()
            f = MagicMock()
            q.filter.return_value = f
            if model.__name__ == "Process":
                f.first.return_value = process
            elif model.__name__ == "ProcessFace":
                f.all.return_value = [pf]
            elif model.__name__ == "Identity":
                f.first.return_value = identity
            return q

        mock_db.query.side_effect = query_side_effect

        response = client.get(f"/processes/{process.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["taskDetails"]["type"] == "recognition"
        assert len(data["faces"]) == 1
        assert data["faces"][0]["status"] == "known"

    def test_get_process_detail_not_found(self, client, mock_db):
        mock_db.query.return_value.filter.return_value.first.return_value = None

        response = client.get(f"/processes/{uuid.uuid4()}")
        assert response.status_code == 404
