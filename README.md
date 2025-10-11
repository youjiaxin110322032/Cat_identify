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
