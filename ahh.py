from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.getcwd(), 'cnn_scoop.h5')
model = load_model(model_path)

# Route for the root URL
@app.route('/')
def home():
    return 'Welcome to the Image Prediction Service!'

# Function to preprocess the image
def preprocess_image(image):
    # Open the image file
    image = Image.open(io.BytesIO(image))
    
    # Resize the image to match the input size of the model (adjust as needed)
    image = image.resize((300, 300))  # Assuming 300x300 is the input size for your model
    image = np.array(image) / 255.0   # Normalize the image pixel values
    
    # Ensure that the image has the correct number of channels (e.g., RGB)
    if image.shape[-1] != 3:
        image = np.stack([image] * 3, axis=-1)  # Convert grayscale to RGB if needed
    
    # Add an extra dimension to match the model's input shape (batch_size, height, width, channels)
    image = np.expand_dims(image, axis=0)
    
    return image

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request contains the 'image' file
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    # Get the image file from the request
    file = request.files['image']
    image = file.read()  # Read the file content
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Use the model to make a prediction
    prediction = model.predict(preprocessed_image)
    
    # Assuming the model outputs a probability distribution (for multi-class classification)
    result = prediction.argmax(axis=-1)  # Get the class with highest probability
    
    return jsonify({'prediction': result.tolist()})

# Route to handle favicon requests
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico')

if __name__ == '__main__':
   app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 1000)), debug=False)
