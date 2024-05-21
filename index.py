from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model
model_path = 'model/model.h5'
model = load_model(model_path)

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize image to match model's expected sizing
    img_array = np.asarray(img)  # Convert PIL image to numpy array
    img_array = img_array / 255.0  # Normalize pixel values to range [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if an image was uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        # Process the uploaded image
        if file:
            image = Image.open(file.stream)
            img_array = preprocess_image(image)
            prediction = model.predict(img_array)

            # Determine the prediction
            result = 'Normal'
            confidence = 1 - prediction[0][0]
            if prediction[0][0] > 0.5:
                result = 'Pneumonia'
                confidence = prediction[0][0]

            return render_template('index.html', result=result, confidence=confidence, image_uploaded=True, image_path=file.filename)

    return render_template('index.html', image_uploaded=False)

if __name__ == '__main__':
    app.run(debug=True)
