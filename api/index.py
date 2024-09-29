# api/index.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import numpy as np
from tensorflow.keras.models import load_model
import os
import time

app = FastAPI()

model_path = "SSRModel.keras"  # อัปโหลดโมเดลไปที่ root directory บน Vercel
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = load_model(model_path)

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Unsupported file type. Only image files are accepted.")

    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
    except IOError:
        raise HTTPException(status_code=400, detail="Invalid image file format")

    processed_image = preprocess_image(image)

    try:
        prediction = model.predict(processed_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model prediction: {str(e)}")

    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence_score = np.max(prediction) * 100

    threshold = 80
    if confidence_score < threshold:
        warning = "Prediction confidence is below threshold"
    else:
        warning = "Prediction is reliable"

    top_n = 3
    top_n_indices = prediction[0].argsort()[-top_n:][::-1]
    top_n_confidences = prediction[0][top_n_indices] * 100

    top_n_predictions = [
        {"class": int(index), "confidence_score": float(f"{conf:.2f}")}
        for index, conf in zip(top_n_indices, top_n_confidences)
    ]

    processing_time = time.time() - start_time

    return {
        "predicted_class": int(predicted_class),
        "confidence_score": f"{confidence_score:.2f}%",
        "warning": warning,
        "top_n_predictions": top_n_predictions,
        "processing_time": processing_time
    }
