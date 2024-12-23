from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import pytesseract
import numpy as np
import io
import pymongo
import base64
from datetime import datetime
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

app = Flask(__name__)

# MongoDB connection
client = pymongo.MongoClient("mongodb+srv://sekhar123:sekhar123@cluster0.dthptno.mongodb.net/")
db = client["image_recognition_db"]
collection = db["images"]

# Load pre-trained model
model = ResNet50(weights='imagenet')

def process_image(image_data):
    # Convert base64 to image
    img = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # Save original image
    img_path = f"uploads/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    img.save(img_path)
    
    # Prepare image for ResNet50
    img_resnet = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img_resnet), axis=0)
    processed_img = preprocess_input(img_array)
    
    # Get image classification
    predictions = model.predict(processed_img)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    
    # Extract text using OCR
    text = pytesseract.image_to_string(img)
    
    return {
        "image_path": img_path,
        "predictions": [
            {"label": label, "confidence": float(confidence)}
            for _, label, confidence in decoded_predictions
        ],
        "extracted_text": text
    }

@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    image_data = base64.b64encode(file.read()).decode()
    
    # Process image
    results = process_image(image_data)
    
    # Store in MongoDB
    document = {
        "timestamp": datetime.now(),
        "image_path": results["image_path"],
        "predictions": results["predictions"],
        "extracted_text": results["extracted_text"]
    }
    collection.insert_one(document)
    
    return jsonify(results)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
