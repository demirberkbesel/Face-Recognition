import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Process, ProcessFace, Identity
from app.schemas import ProcessDetailResponse, ProcessFaceDetail, BoundingBox

router = APIRouter(prefix="/processes", tags=["processes"])


@router.get("/{process_id}", response_model=ProcessDetailResponse)
def get_process_detail(process_id: str, db: Session = Depends(get_db)):
    try:
        pid = uuid.UUID(process_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Geçersiz processId formatı.")

    process = db.query(Process).filter(Process.id == pid).first()
    if not process:
        raise HTTPException(status_code=404, detail="Process ID bulunamadı.")

    pfs = db.query(ProcessFace).filter(ProcessFace.process_id == pid).all()

    faces = []
    for pf in pfs:
        identity = db.query(Identity).filter(Identity.id == pf.identity_id).first()
        face_id = str(identity.id) if identity else str(pf.identity_id)
        faces.append(
            ProcessFaceDetail(
                faceId=face_id,
                status=pf.status,
                confidence=pf.confidence,
                boundingBox=BoundingBox(**pf.bounding_box),
            )
        )

    return ProcessDetailResponse(
        processId=str(process.id),
        timestamp=process.timestamp,
        taskDetails=process.task_details,
        faces=faces,
    )
