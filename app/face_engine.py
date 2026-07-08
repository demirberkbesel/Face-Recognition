import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

from app.config import INSIGHTFACE_MODEL_NAME


class FaceEngine:
    def __init__(self):
        self.app = FaceAnalysis(
            name=INSIGHTFACE_MODEL_NAME,
            root="./models",
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect_faces(self, image: np.ndarray) -> list[dict]:
        faces = self.app.get(image)
        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            results.append({
                "bbox": {
                    "xmin": int(bbox[0]),
                    "ymin": int(bbox[1]),
                    "xmax": int(bbox[2]),
                    "ymax": int(bbox[3]),
                },
                "embedding": face.embedding.astype(np.float32),
                "det_score": float(face.det_score),
            })
        return results
