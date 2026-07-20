import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.models import Identity, FaceEmbedding


MOCK_FACE_ID = uuid.uuid4()
MOCK_PROCESS_ID = uuid.uuid4()

MOCK_BBOX = {"xmin": 10, "ymin": 20, "xmax": 100, "ymax": 150}
MOCK_EMBEDDING = np.random.rand(512).astype(np.float32)


def _ensure_id(obj):
    if isinstance(obj, Identity) and obj.id is None:
        obj.id = uuid.uuid4()
    if isinstance(obj, FaceEmbedding) and obj.id is None:
        obj.id = uuid.uuid4()


@pytest.fixture(autouse=True)
def mock_session_local():
    with patch("app.recognition.SessionLocal") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock


@pytest.fixture(autouse=True)
def mock_save_cropped_face():
    with patch("app.recognition._save_cropped_face") as mock:
        mock.return_value = "images/mock-face.jpg"
        yield mock


@pytest.fixture(autouse=True)
def mock_blur_check():
    """Varsayılan: blur kontrolünü devre dışı bırak (mevcut testler bozulmasın)."""
    with patch("app.recognition.check_blur") as mock:
        mock.return_value = (False, 999.0)
        yield mock


@pytest.fixture
def mock_db():
    db = MagicMock()

    def add_side_effect(obj):
        _ensure_id(obj)

    def flush_side_effect():
        pass

    db.add.side_effect = add_side_effect
    db.flush.side_effect = flush_side_effect
    db.commit = MagicMock()
    db.rollback = MagicMock()
    db.execute = MagicMock()
    db.query = MagicMock()
    return db


@pytest.fixture
def mock_face_engine():
    with patch("app.recognition.engine") as mock_engine:
        mock_engine.detect_faces.return_value = [
            {
                "bbox": MOCK_BBOX,
                "embedding": MOCK_EMBEDDING,
                "det_score": 0.95,
            }
        ]
        yield mock_engine


@pytest.fixture
def mock_empty_face_engine():
    with patch("app.recognition.engine") as mock_engine:
        mock_engine.detect_faces.return_value = []
        yield mock_engine


def make_mock_identity(identity_id=None, status="anonymous", name=None, extra_data=None):
    identity = MagicMock()
    identity.id = identity_id or uuid.uuid4()
    identity.status = status
    identity.name = name
    identity.extra_data = extra_data
    return identity
