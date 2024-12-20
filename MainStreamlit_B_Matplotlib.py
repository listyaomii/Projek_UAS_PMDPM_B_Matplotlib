import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the pre-trained model
model = load_model(r'D:\coolyeah\semester5\ml\tubes_uas\BestModel_GoogleNet_Matplotlib.h5')  # Adjust the path to your model
class_names = ['Busuk', 'Matang', 'Mentah']

# Function to preprocess and classify image
def classify_image(image):
    try:
        # Preprocess the image
        input_image = image.resize((180, 180))  # Resize to match model input
        input_image_array = np.array(input_image)  # Convert to numpy array
        input_image_exp_dim = np.expand_dims(input_image_array, axis=0)  # Add batch dimension

        # Predict using the model
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])  # Apply softmax for probability

        # Get class with highest confidence
        class_idx = np.argmax(result)
        confidence_scores = result.numpy()
        return class_names[class_idx], confidence_scores
    except Exception as e:
        return "Error", str(e)

# Function to create a custom progress bar
def custom_progress_bar(confidence, color1, color2):
    percentage1 = confidence[0] * 100  # Confidence for class 0 (Busuk)
    percentage2 = confidence[1] * 100  # Confidence for class 1 (Matang)
    percentage3 = confidence[2] * 100  # Confidence for class 2 (Mentah)
    progress_html = f"""
    <div style="border: 1px solid #ddd; border-radius: 5px; overflow: hidden; width: 100%; font-size: 14px;">
        <div style="width: {percentage1:.2f}%; background: #FF4136; color: white; text-align: center; height: 24px; float: left;">
            {percentage1:.2f}% Busuk
        </div>
        <div style="width: {percentage2:.2f}%; background: #007BFF; color: white; text-align: center; height: 24px; float: left;">
            {percentage2:.2f}% Matang
        </div>
        <div style="width: {percentage3:.2f}%; background: #2ECC40; color: white; text-align: center; height: 24px; float: left;">
            {percentage3:.2f}% Mentah
        </div>
    </div>
    """
    st.sidebar.markdown(progress_html, unsafe_allow_html=True)

# Streamlit UI
st.title("Prediksi Strawberry")  # 4 digit npm terakhir

# Upload multiple files in the main page
uploaded_files = st.file_uploader("Unggah Gambar (Beberapa diperbolehkan)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Sidebar for prediction button and results
if st.sidebar.button("Prediksi"):
    if uploaded_files:
        st.sidebar.write("### Hasil Prediksi")
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)  # Open the uploaded image

            # Perform prediction
            label, confidence = classify_image(image)

            if label != "Error":
                # Display prediction results
                st.sidebar.write(f"**Nama File:** {uploaded_file.name}")
                st.sidebar.markdown(f"<h4 style='color: #007BFF;'>Prediksi: {label}</h4>", unsafe_allow_html=True)

                # Display confidence scores
                st.sidebar.write("**Confidence:**")
                for i, class_name in enumerate(class_names):
                    st.sidebar.write(f"- {class_name}: {confidence[i] * 100:.2f}%")

                # Display custom progress bar
                custom_progress_bar(confidence, "#FF4136", "#007BFF")

                st.sidebar.write("---")
            else:
                st.sidebar.error(f"Kesalahan saat memproses gambar {uploaded_file.name}: {confidence}")
    else:
        st.sidebar.error("Silakan unggah setidaknya satu gambar untuk diprediksi.")

# Preview images in the main page
if uploaded_files:
    st.write("### Preview Gambar")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"{uploaded_file.name}", use_column_width=True)