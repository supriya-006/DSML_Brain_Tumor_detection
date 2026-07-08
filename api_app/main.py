import io
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "runs" / "detect" / "train-2" / "weights" / "best.pt"

app = FastAPI(title="Brain Tumor Detection API", version="1.0.0")

model: Optional[Any] = None


def load_model() -> Any:
    # Lazy-import Ultralytics to avoid requiring it at server startup.
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        try:
            from ultralytics import YOLO
        except Exception as exc:
            # Surface a clear error so callers can report install instructions.
            raise ImportError(
                "Ultralytics YOLO is not available in the environment. "
                "Install it with `pip install ultralytics` to enable predictions."
            ) from exc

        model = YOLO(str(MODEL_PATH))
    return model


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": str(MODEL_PATH)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    try:
        detector = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        image_bytes = await file.read()
        temp_file.write(image_bytes)
        temp_path = temp_file.name

    try:
        with Image.open(io.BytesIO(image_bytes)).convert("RGB") as image:
            image_array = np.array(image)
            image_height, image_width = image_array.shape[:2]

        results = detector.predict(source=temp_path, conf=0.25, stream=False)
        result = results[0]

        detections = []
        for index, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            detections.append(
                {
                    "id": index + 1,
                    "class": result.names.get(class_id, str(class_id)),
                    "confidence": round(confidence * 100, 2),
                    "bbox": [
                        round(float(box.xyxy[0][0]), 2),
                        round(float(box.xyxy[0][1]), 2),
                        round(float(box.xyxy[0][2]), 2),
                        round(float(box.xyxy[0][3]), 2),
                    ],
                }
            )

        annotated_array = result.plot()
        annotated_image = Image.fromarray(annotated_array)
        buffer = io.BytesIO()
        annotated_image.save(buffer, format="JPEG")
        encoded_image = buffer.getvalue()

        return {
            "filename": file.filename,
            "image_size": {"width": image_width, "height": image_height},
            "detections": detections,
            "annotated_image": encoded_image.hex(),
            "confidence": round(max((d["confidence"] for d in detections), default=0.0), 2),
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_app.main:app", host="0.0.0.0", port=8000, reload=True)
