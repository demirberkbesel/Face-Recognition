import uuid
import logging
from datetime import datetime
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.config import SIMILARITY_THRESHOLD
from app.database import SessionLocal
from app.models import Identity, FaceEmbedding, Process, ProcessFace
from app.face_engine import FaceEngine

logger = logging.getLogger(__name__)

engine = FaceEngine()


def recognize_faces(image: np.ndarray, db: Session) -> dict:
    process_id = uuid.uuid4()

    detected_faces = engine.detect_faces(image)
    face_results = []
    identity_ids_in_process = []

    for det in detected_faces:
        embedding = det["embedding"]
        bbox = det["bbox"]

        matched_identity = _find_nearest_match(embedding, db)

        if matched_identity and matched_identity["distance"] < (1.0 - SIMILARITY_THRESHOLD):
            identity_id = matched_identity["identity_id"]
            identity = db.query(Identity).filter(Identity.id == identity_id).first()
            confidence = round(1.0 - matched_identity["distance"], 4)
            status = identity.status
            name = identity.name
            metadata = identity.extra_data
        else:
            identity = Identity(
                id=uuid.uuid4(),
                status="anonymous",
                name=None,
                metadata=None,
            )
            db.add(identity)
            db.flush()

            new_embedding = FaceEmbedding(
                identity_id=identity.id,
                embedding=embedding.tolist(),
            )
            db.add(new_embedding)
            db.flush()

            identity_id = identity.id
            confidence = 0.0
            status = "new_anonymous"
            name = None
            metadata = None

        identity_ids_in_process.append({
            "identity_id": identity_id,
            "status": status,
            "confidence": confidence,
            "bbox": bbox,
            "name": name,
            "metadata": metadata,
        })

        face_results.append({
            "faceId": str(identity_id),
            "status": status,
            "name": name,
            "metadata": metadata,
            "boundingBox": bbox,
            "confidence": confidence,
        })

    db.commit()

    _log_process_background(process_id, detected_faces, identity_ids_in_process)

    return {
        "processId": str(process_id),
        "faceCount": len(face_results),
        "faces": face_results,
    }


def enroll_face(image: np.ndarray, name: str, metadata: Optional[dict], db: Session) -> dict:
    detected_faces = engine.detect_faces(image)
    if not detected_faces:
        raise ValueError("Görselde yüz bulunamadı.")

    primary_face = detected_faces[0]
    embedding = primary_face["embedding"]

    identity = Identity(
        status="known",
        name=name,
        metadata=metadata,
    )
    db.add(identity)
    db.flush()

    face_embedding = FaceEmbedding(
        identity_id=identity.id,
        embedding=embedding.tolist(),
    )
    db.add(face_embedding)
    db.flush()

    db.commit()

    return {
        "faceId": str(identity.id),
        "status": "known",
        "name": name,
        "metadata": metadata,
    }


def enroll_existing_face(face_id: uuid.UUID, name: str, metadata: Optional[dict], db: Session) -> dict:
    identity = db.query(Identity).filter(Identity.id == face_id).first()
    if not identity:
        raise ValueError("Face ID bulunamadı.")

    identity.status = "known"
    identity.name = name
    identity.extra_data = metadata
    db.commit()

    return {
        "faceId": str(identity.id),
        "status": "known",
        "name": name,
        "metadata": metadata,
    }


def _find_nearest_match(embedding: np.ndarray, db: Session) -> Optional[dict]:
    embedding_list = embedding.tolist()
    result = db.execute(
        text("""
            SELECT identity_id, (embedding <=> :emb) as distance
            FROM face_embeddings
            ORDER BY distance ASC
            LIMIT 1
        """),
        {"emb": embedding_list},
    ).fetchone()

    if result:
        return {"identity_id": result[0], "distance": result[1]}
    return None


def _log_process_background(
    process_id: uuid.UUID,
    detected_faces: list,
    identity_ids_in_process: list,
):
    try:
        log_db = SessionLocal()
        process = Process(
            id=process_id,
            timestamp=datetime.utcnow(),
            task_details={
                "type": "recognition",
                "faceCount": len(detected_faces),
            },
        )
        log_db.add(process)

        for entry in identity_ids_in_process:
            pf = ProcessFace(
                process_id=process_id,
                identity_id=entry["identity_id"],
                status=entry["status"],
                confidence=entry["confidence"],
                bounding_box=entry["bbox"],
            )
            log_db.add(pf)

        log_db.commit()
        log_db.close()
    except Exception as e:
        logger.error(f"Log kaydı başarısız oldu (process {process_id}): {e}")
