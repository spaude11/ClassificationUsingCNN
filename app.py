import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize image to match model's expected sizing
    img_array = np.asarray(img)  # Convert PIL image to numpy array
    img_array = img_array / 255.0  # Normalize pixel values to range [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to classify the uploaded image
def classify_image(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    return prediction

# Load your model
model = load_model('cnn.h5')  # Update with your model file path

# Set up the Streamlit app layout
st.title('Chest X-ray Image Classifier')
st.write('Upload a chest X-ray image for classification.')

# Allow user to upload an image
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

# If user uploaded an image
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Classify the uploaded image
    with st.spinner('Classifying...'):
        prediction = classify_image(image)

    # Display the classification results
    if prediction[0][0] > 0.5:
        st.write('Prediction: Pneumonia')
        st.write('Confidence:', prediction[0][0])
    else:
        st.write('Prediction: Normal')
        st.write('Confidence:', 1 - prediction[0][0])
