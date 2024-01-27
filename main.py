from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import uvicorn
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to the actual origin of your HTML page in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Adjust the endpoint according to your TensorFlow Serving setup
endpoint = "http://localhost:8001/v1/models/model:predict"

MODEL_PATH = "C:/Users/User/Desktop/agrihub/model/1"  # Update the path accordingly
MODEL = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get('/ping')
async def ping():
    return "hello"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    # Ensure the image has 3 channels (drop the alpha channel if present)
    image = image.convert('RGB')
    # Resize the image to (256, 256)
    resized_image = image.resize((256, 256))
    image_array = np.array(resized_image)
    return image_array
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and process the file
        bytes_data = await file.read()
        image_array = read_file_as_image(bytes_data)
        img_batch = np.expand_dims(image_array, 0)

        # Check if the image is not green (non-leaf)
        predictions = MODEL.predict(img_batch)
        confidence = np.max(predictions[0])
        threshold  = 0.5
        if confidence >= threshold:
            # Include the raw predictions in the response
            return {
                'class': CLASS_NAMES[np.argmax(predictions[0])],
                'confidence': float(confidence),
                'raw_predictions': predictions.tolist(),  # Convert to a list for JSON serialization
                'message': f'it\'s potato leaf which is {CLASS_NAMES[np.argmax(predictions[0])]}'   
            }   
        else:
            return {
                'class': None,
                'confidence': None,
                'raw_predictions': predictions.tolist(),
                'message': 'Not a leaf (insufficient confidence)'
            }
    except ValueError as ve:
        return {
            'class': None,
            'confidence': None,
            'raw_predictions': None,
            'message': str(ve)
        }



if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8001)




