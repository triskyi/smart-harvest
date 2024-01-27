from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import uvicorn
import requests

app = FastAPI()

# Adjust the endpoint according to your TensorFlow Serving setup
endpoint = "http://localhost:8605/v1/models/model:predict"

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get('/ping')
async def ping():
    return "hello"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and process the file
    bytes_data = await file.read()
    image_array = read_file_as_image(bytes_data)
    img_batch = np.expand_dims(image_array, 0)

    # Prepare data for prediction
    json_data = {
        "instances": img_batch.tolist()
    }

    # Make a prediction request to the TensorFlow Serving model
    response = requests.post(endpoint, json=json_data)

    # Process the prediction result
    prediction = np.array(response.json()["predictions"][0])
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return {
        "class": predicted_class,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8003)
