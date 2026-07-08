import insightface
from insightface.model_zoo import get_model

if __name__ == "__main__":
    app = insightface.app.FaceAnalysis(name="buffalo_l", root="./models")
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("InsightFace modelleri başarıyla indirildi ve ./models/ altına kaydedildi.")
