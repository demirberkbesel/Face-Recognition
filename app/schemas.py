from typing import Optional
from uuid import UUID
from datetime import datetime

from pydantic import BaseModel


class BoundingBox(BaseModel):
    xmin: int
    ymin: int
    xmax: int
    ymax: int


class FaceResult(BaseModel):
    faceId: str
    status: str
    name: Optional[str] = None
    metadata: Optional[dict] = None
    boundingBox: BoundingBox
    confidence: float
    qualityWarning: Optional[str] = None


class RecognizeResponse(BaseModel):
    processId: str
    faceCount: int
    faces: list[FaceResult]


class EnrollRequestNew(BaseModel):
    name: str
    metadata: Optional[dict] = None


class EnrollRequestExisting(BaseModel):
    faceId: str
    name: str
    metadata: Optional[dict] = None


class EnrollResponse(BaseModel):
    faceId: str
    status: str
    name: Optional[str] = None
    metadata: Optional[dict] = None


class FaceDetailResponse(BaseModel):
    faceId: str
    status: str
    name: Optional[str] = None
    metadata: Optional[dict] = None
    imagePath: Optional[str] = None


class DeleteResponse(BaseModel):
    message: str
    faceId: str


class HistoryItem(BaseModel):
    processId: str
    timestamp: datetime
    status: str
    confidence: float
    boundingBox: BoundingBox


class FaceHistoryResponse(BaseModel):
    faceId: str
    history: list[HistoryItem]


class ProcessFaceDetail(BaseModel):
    faceId: str
    status: str
    confidence: float
    boundingBox: BoundingBox


class ProcessDetailResponse(BaseModel):
    processId: str
    timestamp: datetime
    taskDetails: dict
    faces: list[ProcessFaceDetail]
