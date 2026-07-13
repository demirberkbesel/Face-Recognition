import uuid
import os
import json

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Identity, FaceEmbedding, ProcessFace, Process
from app.schemas import (
    RecognizeResponse,
    EnrollResponse,
    FaceDetailResponse,
    DeleteResponse,
    FaceHistoryResponse,
    HistoryItem,
    BoundingBox,
)
from app.recognition import recognize_faces, enroll_face, enroll_existing_face

router = APIRouter(prefix="/faces", tags=["faces"])


@router.post("/recognize", response_model=RecognizeResponse)
def recognize(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail="Desteklenmeyen dosya formatı. JPEG veya PNG gönderin.",
        )

    image_bytes = file.file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Boş dosya gönderilemez.")

    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Görsel okunamadı veya bozuk.")

    result = recognize_faces(image, db)
    return result


@router.post("/enroll", response_model=EnrollResponse)
def enroll(
    file: UploadFile = File(None),
    name: str = Form(...),
    metadata: str = Form("null"),
    face_id: str = Form(None),
    db: Session = Depends(get_db),
):
    parsed_metadata = json.loads(metadata) if metadata != "null" else None

    if face_id:
        try:
            fid = uuid.UUID(face_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Geçersiz faceId formatı.")
        return enroll_existing_face(fid, name, parsed_metadata, db)

    if not file:
        raise HTTPException(
            status_code=400,
            detail="Yeni kayıt için görsel gereklidir.",
        )

    image_bytes = file.file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Boş dosya gönderilemez.")

    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Görsel okunamadı.")

    try:
        result = enroll_face(image, name, parsed_metadata, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return result


@router.get("/{face_id}", response_model=FaceDetailResponse)
def get_face_detail(face_id: str, db: Session = Depends(get_db)):
    try:
        fid = uuid.UUID(face_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Geçersiz faceId formatı.")

    identity = db.query(Identity).filter(Identity.id == fid).first()
    if not identity:
        raise HTTPException(status_code=404, detail="Face ID bulunamadı.")

    latest_embedding = (
        db.query(FaceEmbedding)
        .filter(FaceEmbedding.identity_id == fid)
        .order_by(FaceEmbedding.created_at.desc())
        .first()
    )

    current_path = latest_embedding.image_path if latest_embedding else None
    image_url = f"/faces/{face_id}/image" if current_path else None

    return FaceDetailResponse(
        faceId=str(identity.id),
        status=identity.status,
        name=identity.name,
        metadata=identity.extra_data,
        imagePath=image_url,
    )


@router.delete("/{face_id}", response_model=DeleteResponse)
def delete_face(face_id: str, db: Session = Depends(get_db)):
    try:
        fid = uuid.UUID(face_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Geçersiz faceId formatı.")

    identity = db.query(Identity).filter(Identity.id == fid).first()
    if not identity:
        raise HTTPException(status_code=404, detail="Face ID bulunamadı.")

    db.delete(identity)
    db.commit()

    return DeleteResponse(
        message="Face ID ve bağlı vektörler başarıyla silindi.",
        faceId=face_id,
    )


@router.get("/{face_id}/history", response_model=FaceHistoryResponse)
def get_face_history(face_id: str, db: Session = Depends(get_db)):
    try:
        fid = uuid.UUID(face_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Geçersiz faceId formatı.")

    identity = db.query(Identity).filter(Identity.id == fid).first()
    if not identity:
        raise HTTPException(status_code=404, detail="Face ID bulunamadı.")

    pfs = (
        db.query(ProcessFace)
        .filter(ProcessFace.identity_id == fid)
        .order_by(ProcessFace.created_at.desc())
        .all()
    )

    history = []
    for pf in pfs:
        process = db.query(Process).filter(Process.id == pf.process_id).first()
        history.append(
            HistoryItem(
                processId=str(pf.process_id),
                timestamp=process.timestamp if process else pf.created_at,
                status=pf.status,
                confidence=pf.confidence,
                boundingBox=BoundingBox(**pf.bounding_box),
            )
        )

    return FaceHistoryResponse(faceId=face_id, history=history)


@router.get("/{face_id}/image")
def get_face_image(face_id: str, db: Session = Depends(get_db)):
    try:
        fid = uuid.UUID(face_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Geçersiz faceId formatı.")

    latest_embedding = (
        db.query(FaceEmbedding)
        .filter(FaceEmbedding.identity_id == fid)
        .order_by(FaceEmbedding.created_at.desc())
        .first()
    )
    if not latest_embedding or not latest_embedding.image_path:
        raise HTTPException(status_code=404, detail="Bu yüz için kayıtlı resim bulunamadı.")

    if not os.path.exists(latest_embedding.image_path):
        raise HTTPException(status_code=404, detail="Resim dosyası diskte bulunamadı.")

    return FileResponse(latest_embedding.image_path, media_type="image/jpeg")
