import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import traceback

# Load pre-trained model and data
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Add CSS for background and animation
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to right, #a8caba, #5d4157);
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Outfit Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        # Create the 'uploads' directory if it doesn't exist
        os.makedirs('uploads', exist_ok=True)
        # Save the uploaded file
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        st.error("An error occurred while saving the file.")
        st.error(traceback.format_exc())  # Print the traceback
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# File upload
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption='Uploaded Image', use_column_width=True)

        # Feature extraction
        st.write("Extracting features...")
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)

        # Recommendation
        st.write("Performing recommendation...")
        indices = recommend(features, feature_list)

        # Show recommendations
        columns = st.columns(5)
        if indices is not None and len(indices) > 0:
            num_recommendations = min(len(indices[0]), 5)  # Limit to 5 recommendations if more than 5 are available
            for i in range(num_recommendations):
                if indices[0][i] < len(filenames):
                    image_path = filenames[indices[0][i]]
                    try:
                        with columns[i]:
                            recommended_image = Image.open(image_path)
                            st.image(recommended_image, caption=f'Recommended Image {i+1}', use_column_width=True)
                    except Exception as e:
                        st.error("Error loading image:", e)
                else:
                    st.write("Index", indices[0][i], "is out of range.")
        else:
            st.write("No recommendations found.")
