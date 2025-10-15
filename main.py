# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np, io
from PIL import Image

# 你的辨識模組
from catfaces_demo import load_model, detect_cat_faces, face_to_feature, K, UNKNOWN_THRESHOLD

app = FastAPI(title="Cat Face ID API", version="1.1")

# ===== CORS =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://youjiaxin110322032.github.io",  # ✅ GitHub Pages 前端
    ],
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

@app.get("/labels")
def labels():
    """回傳目前模型裡的已知貓名（檢查/顯示用）"""
    return {"count": len(id2name), "labels": [id2name[i] for i in sorted(id2name.keys())]}

@app.post("/reload")
def reload_model():
    """若替換了 cat_knn.pkl / labels.json，可熱重載"""
    global knn, id2name
    knn, id2name = load_model()
    return {"reloaded": True, "count": len(id2name)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        # 讀圖 + 轉 BGR（OpenCV 習慣）
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
