from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os

app = FastAPI(title="Ship Detection API")

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model lazily
model = None

def get_model():
    global model
    if model is None:
        model_path = "ship_model_final.keras"
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
        else:
            print("Model not found. Please train it first.")
    return model

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    m = get_model()
    if m is None:
        return {"error": "Model not loaded. Train the model first."}
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image = image.resize((80, 80))
        img_array = np.array(image)
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = m.predict(img_array)
        score = float(prediction[0][0])
        
        # Classify
        is_ship = score >= 0.5
        
        return {
            "prediction": int(is_ship),
            "prob_ship": float(score),
            "prob_no_ship": float(1 - score),
            "filename": file.filename
        }
    except Exception as e:
        return {"error": str(e)}

# Serve frontend files
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def read_root():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Ship Detection API is running. Frontend not found."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
