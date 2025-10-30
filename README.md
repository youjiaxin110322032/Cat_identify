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

# ⚙️ 專案啟動注意事項

## 🐍 Python 版本
請務必使用 **Python 3.12.10**（其他版本可能會發生相容性問題）。

## 🧩 建立虛擬環境
專案啟動前，請先建立虛擬環境（venv）：

### Windows PowerShell：
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

#請在虛擬環境啟用後執行：pip install -r requirements.txt