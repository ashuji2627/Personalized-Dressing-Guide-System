import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import base64


# Page configuration


# Assuming you already have these variables
st.set_page_config(page_title="Personalized Dressing Guide", layout="wide")

# Background image setup
def set_bg_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_bg_image("Background_image.png")  # Replace with your path if different

# Load features and model

# feature_list = np.array(joblib.load(open('embeddings.joblib', 'rb')))
# filenames = joblib.load(open('filenames.joblib', 'rb'))

# with open("embeddings.pkl", "wb") as f:
#     pickle.dump(feature_list, f)

# with open("filenames.pkl", "wb") as f:
#     pickle.dump(filenames, f)

feature_list = np.array(pickle.load(open('embeddings.pkl', 'wb')))
filenames = pickle.load(open('filenames.pkl', 'wb'))

resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet.trainable = False
model = tf.keras.Sequential([
    resnet,
    GlobalMaxPooling2D()
])

# Custom UI styling
st.markdown("""
    <style>
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 30px;
        margin: 20px auto;
        max-width: 850px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.3);
    }
    h1 {
        text-align: center;
        font-size: 2.5rem;
        color: #ffffff;
        text-shadow: 1px 1px 2px #00000055;
    }
    .stFileUploader label {
        color: white;
        font-size: 1rem;
        background-color: rgba(255, 255, 255, 0.1);
        padding: 8px 12px;
        border-radius: 6px;
        display: inline-block;
    }
    .stImage img {
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.4);
    }
    .outfit-caption {
        background-color: rgba(0, 0, 0, 0.6);
        color: white;
        padding: 4px 8px;
        border-radius: 6px;
        text-align: center;
        font-weight: bold;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Title block
st.markdown('<div class="glass-card"><h1>Personalized Dressing Guide System</h1>', unsafe_allow_html=True)

# Upload section
st.subheader("📤 Upload Your Outfit Image")
uploaded_file = st.file_uploader("Choose an image file (jpg, png)", type=["jpg", "jpeg", "png"])

# Save uploaded file
def save_uploaded_file(file):
    try:
        os.makedirs('uploads', exist_ok=True)
        filepath = os.path.join('uploads', file.name)
        with open(filepath, 'wb') as f:
            f.write(file.getbuffer())
        return filepath
    except Exception as e:
        st.error(f"File Save Error: {e}")
        return None

# Feature extraction
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed = preprocess_input(expanded_img)
    result = model.predict(preprocessed).flatten()
    normalized = result / norm(result)
    return normalized

# Recommendation logic
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# Main workflow
if uploaded_file is not None:
    saved_path = save_uploaded_file(uploaded_file)
    if saved_path:
        st.markdown('</div>', unsafe_allow_html=True)  # Close title card

        # Uploaded image preview
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("👗 Uploaded Image Preview")
        st.image(saved_path, width=300)

        # Feature extraction and recommendation
        features = feature_extraction(saved_path, model)
        indices = recommend(features, feature_list)

        # Show recommendations
        st.subheader("💡 Recommended Outfit Matches")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            with col:
                img_path = filenames[indices[0][i]]
                try:
                    img = Image.open(img_path).convert("RGB")
                    col.image(img, use_container_width=True)
                    col.markdown(f'<div class="outfit-caption">Outfit {i+1}</div>', unsafe_allow_html=True)
                except Exception as e:
                    col.error(f"Error loading image: {e}")

        st.markdown('</div>', unsafe_allow_html=True)  # Close recommendations card
    else:
        st.error("⚠️ Error saving the uploaded image.")
