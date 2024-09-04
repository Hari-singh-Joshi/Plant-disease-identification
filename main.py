from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model from the .h5 file
MODEL = tf.keras.models.load_model("../models/potatoes.h5")

# Define the class names
CLASS_NAMES = [
    "Brown Spot",       # 0
    "Healthy Leaf",     # 1
    "Leaf Blast",       # 2
    "Leaf Blight",      # 3
    "Leaf Smut",        # 4
]


@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive"}

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    # Convert image to RGB if it's not already in that format
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Resize image to the size expected by the model, e.g., (224, 224)
    image = image.resize((256, 256))
    image_array = np.array(image)
    # Normalize the image (e.g., to the range [0, 1] if necessary)
    image_array = image_array / 255.0
    return image_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess the file
        image = read_file_as_image(await file.read())

        # Ensure the image is in the correct shape for the model
        img_batch = np.expand_dims(image, axis=0)  # Add batch dimension

        # Perform prediction
        predictions = MODEL.predict(img_batch)

        # Determine the predicted class and confidence
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        print(predicted_class, confidence)

        return {
            "filename": file.filename,
            "class": predicted_class,
            "confidence": float(confidence)
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
