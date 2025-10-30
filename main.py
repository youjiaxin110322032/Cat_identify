# main.py
import os
import io
import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from catfaces_demo import (
    load_model, detect_cat_faces, face_to_feature,
    K, UNKNOWN_THRESHOLD
)

ENV = os.getenv("ENV", "local")

app = FastAPI(title="Cat Face ID API", version="1.1")

# CORS（開發期先全開，上線可鎖定你的網域）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# 資料庫：依 ENV 切換
# ---------------------------
if ENV == "vercel":
    # 雲端（例如 Vercel）：由你的 libsql_db 封裝對 Turso/LibSQL 的查詢
    from libsql_db import query_all_cats  # 需自行提供此模組與 function

    @app.get("/cats")
    def list_cats_cloud():
        """雲端模式：直接呼叫 LibSQL 的查詢函式"""
        return query_all_cats()

else:
    # 本地：SQLAlchemy + SQLite
    from sqlalchemy.orm import Session
    from database import get_db, Base, engine
    from models import Cat, Household, Caretaker

    # 只在本地建立資料表
    Base.metadata.create_all(bind=engine)

    @app.get("/cats")
    def list_cats_local(db: Session = Depends(get_db)):
        rows = db.query(Cat).all()
        return [
            {
                "id": c.id,
                "name": c.name,
                "sex": c.sex,
                "coat": c.coat,
                "ear_tip": bool(c.ear_tip),
                "household": c.household.name if getattr(c, "household", None) else None,
                "caretakers": [k.name for k in getattr(c, "caretakers", [])],
            }
            for c in rows
        ]

# ---------------------------
# 基本健康檢查
# ---------------------------
@app.get("/")
def root():
    return {"ok": True, "env": ENV, "hint": "GET /cats, POST /predict"}

@app.get("/ping")
def ping():
    return {"pong": True}

# ---------------------------
# 貓臉辨識：模型載入 & API
# ---------------------------
# 啟動時載入模型（需與 main.py 同資料夾有 cat_knn.pkl / labels.json）
knn, id2name = load_model()

@app.get("/labels")
def labels():
    return {"count": len(id2name), "labels": [id2name[i] for i in sorted(id2name.keys())]}

@app.post("/reload")
def reload_model():
    global knn, id2name
    knn, id2name = load_model()
    return {"reloaded": True, "count": len(id2name)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        # PIL (RGB) -> OpenCV (BGR)
        img = np.array(img)[:, :, ::-1]

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

        H, W = img.shape[:2]
        return {"width": W, "height": H, "boxes": boxes}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image or server error: {e}")


# ---------------------------cd C:\Users\11032\Desktop\cats
# 本地啟動（開發）
# uvicorn main:app --reload
# ---------------------------
