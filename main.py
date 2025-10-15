# main.py
import os
import io
import logging
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# 你的辨識模組
from catfaces_demo import (
    load_model,
    detect_cat_faces,
    face_to_feature,
    K as K_TRAINED,
    UNKNOWN_THRESHOLD as THRESH_TRAINED,
)

# ===== Logging =====
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("cat-face-id")

app = FastAPI(title="Cat Face ID API", version="1.2")

# ===== CORS =====
# 本機除錯想全開 → 改成 allow_origins=["*"]
ALLOW_ORIGINS = [
    "https://youjiaxin110322032.github.io",   # 你的 GitHub Pages
    "http://localhost",
    "http://127.0.0.1",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ← 先全開，待會驗證通了再收斂
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ===== 啟動載模型 =====
try:
    knn, id2name = load_model()
    log.info("Model loaded. labels=%d, K(trained)=%d, THRESH(trained)=%.3f",
             len(id2name), K_TRAINED, THRESH_TRAINED)
except Exception as e:
    log.exception("Failed to load model at startup: %s", e)
    raise

def get_threshold() -> float:
    """支援用環境變數覆蓋 Unknown 門檻。"""
    envv = os.getenv("UNKNOWN_THRESHOLD")
    if envv:
        try:
            return float(envv)
        except ValueError:
            log.warning("UNKNOWN_THRESHOLD=%r 不是有效數字，改用訓練時的預設 %.3f", envv, THRESH_TRAINED)
    return THRESH_TRAINED

@app.get("/")
def root():
    return {"status": "ok", "message": "Cat Face ID API running."}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/config")
def config():
    return {
        "labels_count": len(id2name),
        "labels": [id2name[i] for i in sorted(id2name.keys())],
        "K_trained": K_TRAINED,
        "UNKNOWN_THRESHOLD_in_effect": get_threshold(),
    }

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
    log.info("Model reloaded. labels=%d", len(id2name))
    return {"reloaded": True, "count": len(id2name), "labels": [id2name[i] for i in sorted(id2name.keys())]}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        # 讀圖 + 轉 BGR（OpenCV 習慣）
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img = np.array(img)[:, :, ::-1]

        H, W = img.shape[:2]
        faces = detect_cat_faces(img)
        log.info("[DEBUG] faces=%d (W=%d H=%d)", len(faces), W, H)

        boxes = []
        thr = get_threshold()

        for (x, y, w, h) in faces:
            feat = face_to_feature(img, (x, y, w, h)).reshape(1, -1)
            pred = knn.predict(feat)[0]
            distances, _ = knn.kneighbors(feat, n_neighbors=K_TRAINED, return_distance=True)
            proba = float(np.clip((1 - distances[0]).mean(), 0.0, 1.0))
            name = id2name.get(int(pred), "Unknown")
            if proba < thr:
                name = "Unknown"

            log.info("[DEBUG] box=(%d,%d,%d,%d) pred=%s proba=%.3f thr=%.3f",
                     x, y, w, h, name, proba, thr)

            boxes.append({
                "x": int(x), "y": int(y), "w": int(w), "h": int(h),
                "name": name, "proba": proba
            })

        return {"width": W, "height": H, "boxes": boxes}

    except Exception as e:
        log.exception("predict error: %s", e)
        raise HTTPException(status_code=400, detail=f"Invalid image or server error: {e}")
