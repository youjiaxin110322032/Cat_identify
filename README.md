# 🐱 Cat Face Recognition API

This project is a simple FastAPI-based service for identifying cats using OpenCV and KNN.

## Features
- Upload an image to `/predict`
- Detect cat faces using `haarcascade_frontalcatface.xml`
- Classify cat names using trained KNN model (`cat_knn.pkl`)

## Run locally
```bash
pip install -r requirements.txt
uvicorn main:app --reload

[開啟我的相機辨識網站](https://youjiaxin110322032.github.io/Cat-Face-ID-API/)
[查看 FastAPI 後端 /docs](https://api-server-1-dfq6.onrender.com/docs)
