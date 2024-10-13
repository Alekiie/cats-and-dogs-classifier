import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# import cv2
import cv2

# Set the image dimensions
IMAGE_DIM = (256, 256)

# Set up the Streamlit page
st.set_page_config(page_title="🏡 Cat-Dog Classifier", layout="centered")
st.title("Welcome to Cat-Dog Classifier")

st.markdown("---")
st.subheader("Upload an image of a cat or dog and see the prediction!")

# Upload the image
uploaded_file = st.file_uploader(label="", type=["jpg", "png"])
st.markdown("---")

# Load your pre-trained model
model = load_model('cats_vs_dogs_model.h5', compile=False)
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

# Function to preprocess and predict the class of the image
def predict_image(img):
    # Convert the PIL image to an array and resize it
    img = cv2.resize(IMAGE_DIM)
    img_array = image.img_to_array(img)
    
    # Expand dimensions to add batch size
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess the image for MobileNetV2
    img_array = preprocess_input(img_array)
    
    # Make prediction
    prediction = model.predict(img_array)
    
    if yhat > 0.5: 
        print(f'Predicted class is a Sad Baby')
    else:
        print(f'Predicted class is a Happy Baby')

# If an image is uploaded
if uploaded_file is not None:
    # Open the image
    img = Image.open(uploaded_file)
    
    # Make a prediction
    pred = predict_image(img)
    
    # Display the prediction
    st.subheader(f"Predicted: {pred}!")
    
    # Show the uploaded image
    st.image(img, use_column_width=True)

# Add a footer watermark centered at the bottom
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 50%;
        bottom: 0;
        transform: translateX(-50%);
        width: 100%;
        text-align: center;
        padding: 10px 0;
        background-color: #f1f1f1;
        color: #555;
    }
    </style>
    <div class="footer">
        <p>Made with ❤️ by Alexander</p>
        <p>See more at <a href="https://github.com/alekiie" target="_blank">https://github.com/alekiie</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
