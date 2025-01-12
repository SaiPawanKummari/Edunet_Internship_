import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import io
import pandas as pd

# Load the pre-trained model globally
model = tf.keras.models.load_model(r"C:\Users\CW\Downloads\plant_disease_detetcion.keras")

def model_predict(image_path):
    """Predict the disease from the image path."""
    img = cv2.imread(image_path)
    H, W, C = 224, 224, 3
    img = cv2.resize(img, (H, W))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)
    img = img.astype("float32")
    img = img / 255.0
    img = img.reshape(1, H, W, C)

    prediction = np.argmax(model.predict(img), axis=-1)[0]

    return prediction

# Streamlit app interface
st.title("AgriShield: AI-Powered Plant Disease Detection")

# Custom CSS for background image
st.markdown(
    """
    <style>
    .main {
        background-image: url('C:\\Users\\CW\\Downloads\\background_img.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        opacity: 0.9;
    }
    .reportview-container .main .block-container{
        background-color: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
        
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Home", "Disease Detection", "Project Documentation", "Model Information"])

if app_mode == "Home":
    st.header("Welcome to AgriShield")
    st.write("""
        AgriShield leverages AI to provide fast and accurate plant disease detection.
        Upload an image of a plant leaf, and our AI model will predict if the plant is healthy or has a disease.
    """)
    st.image(r"C:\Users\CW\Downloads\Pdd_img.jpg", caption='Example Plant Image', use_container_width=True)

elif app_mode == "Disease Detection":
    st.header("AI-Powered Plant Disease Detection")
    test_image = st.file_uploader("Choose an Image:")

    if test_image is not None:
        save_path = os.path.join(os.getcwd(), test_image.name)

        with open(save_path, "wb") as f:
            f.write(test_image.getbuffer())

        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)

        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction")
            result_index = model_predict(save_path)

            class_name = ['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
        'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)__Common_rust','Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy','Grape___Black_rot','Grape__Esca(Black_Measles)','Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
        'Grape___healthy','Orange__Haunglongbing(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy'
        'Pepper,bell__Bacterial_spot','Pepper,bell__healthy','Potato___Early_blight','Potato___Late_blight',
        'Potato___healthy','Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch',
        'Strawberry___healthy','Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy']

            st.success(f"Model is predicting it's a {class_name[result_index]}")

elif app_mode == "Project Documentation":
    st.header("Project Documentation")
    st.write("Upload or view documentation related to the project.")
    doc_file = st.file_uploader("Upload a Document:", type=['pdf', 'docx', 'pptx'])
    if doc_file is not None:
        with open(os.path.join(os.getcwd(), doc_file.name), "wb") as f:
            f.write(doc_file.getbuffer())
        st.success("Document uploaded successfully!")

elif app_mode == "Model Information":
    st.header("Model Information")
    st.write("This section provides details about the model architecture and performance.")
    st.text("Model Summary:")
    model.summary(print_fn=lambda x: st.text(x))