from flask import Flask, request, jsonify
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
    # Open the image file and ensure it is in RGB format
    image = Image.open(io.BytesIO(image)).convert('RGB')
    
    # Resize the image to match the input size of the model
    image = image.resize((300, 300))  # Adjust size if necessary
    image = np.array(image) / 255.0   # Normalize the image pixel values
    
    # Add an extra dimension to match the model's input shape (batch_size, height, width, channels)
    image = np.expand_dims(image, axis=0)
    
    return image

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request contains the 'image' file
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    try:
        # Get the image file from the request
        file = request.files['image']
        image = file.read()  # Read the file content
        
        # Preprocess the image
        preprocessed_image = preprocess_image(image)
        
        # Use the model to make a prediction
        prediction = model.predict(preprocessed_image)
        
        # Assuming the model outputs a probability distribution (for multi-class classification)
        result = prediction.argmax(axis=-1)  # Get the class with the highest probability
        
        return jsonify({'prediction': result.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 1000)), debug=False)
