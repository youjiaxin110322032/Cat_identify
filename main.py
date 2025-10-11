# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np, cv2, io
from PIL import Image

# 從你的辨識模組匯入
from catfaces_demo import load_model, detect_cat_faces, face_to_feature, K, UNKNOWN_THRESHOLD

app = FastAPI(title="Cat Face ID API", version="1.0")

# ===== CORS 設定 =====
# 開發中：先用 ["*"] 放行全部（上線請改成你的前端網域）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://http://127.0.0.1:8000/predict"],  # 例如改成 ["http://127.0.0.1:5500", "http://localhost:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 啟動時載模型
knn, id2name = load_model()

@app.get("/")
def root():
    return {"status": "ok", "message": "Cat Face ID API running."}

@app.get("/ping")
def ping():
    return {"pong": True}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        # 讀圖 + 轉 BGR（給 OpenCV）
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img = np.array(img)[:, :, ::-1]

        H, W = img.shape[:2]
        faces = detect_cat_faces(img)
        boxes = []

        for (x, y, w, h) in faces:
            feat = face_to_feature(img, (x, y, w, h)).reshape(1, -1)
            pred = knn.predict(feat)[0]
            distances, _ = knn.kneighbors(feat, n_neighbors=K, return_distance=True)
            proba = float(np.clip((1 - distances[0]).mean(), 0.0, 1.0))

            name = id2name.get(int(pred), "Unknown")
            if proba < UNKNOWN_THRESHOLD:
                name = "Unknown"

            boxes.append({
                "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "name": name, "proba": proba
            })

        return {"width": W, "height": H, "boxes": boxes}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image or server error: {e}")
