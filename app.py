
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="AI Glasses Recommender",
    page_icon="👓",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #f5f5f5;
}
h1 {
    color: #222222;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: #666666;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

st.title("👓 AI Glasses Recommendation")
st.markdown(
    '<p class="subtitle">Discover eyeglass frames that suit your face shape.</p>',
    unsafe_allow_html=True
)

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("face_shape_model.keras")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.info("Place face_shape_model.keras in the same folder as app.py.")
    st.stop()

# Face shape labels
# IMPORTANT: Change these labels to match your model's training class order.
class_names = ["Heart", "Oblong", "Oval", "Round", "Square"]

# Frame recommendations
recommendations = {
    "Heart": {
        "frames": ["Rimless", "Light-coloured frames", "Oval frames"],
        "colours": ["Rose Gold", "Light Brown", "Silver"]
    },
    "Oblong": {
        "frames": ["Oversized frames", "Wayfarer", "Tall rectangular frames"],
        "colours": ["Black", "Brown", "Dark Blue"]
    },
    "Oval": {
        "frames": ["Cat-eye", "Wayfarer", "Square frames"],
        "colours": ["Black", "Gold", "Tortoiseshell"]
    },
    "Round": {
        "frames": ["Rectangular", "Square", "Geometric frames"],
        "colours": ["Black", "Navy Blue", "Dark Brown"]
    },
    "Square": {
        "frames": ["Round", "Oval", "Aviator frames"],
        "colours": ["Gold", "Silver", "Transparent"]
    }
}

# Upload image
uploaded_file = st.file_uploader(
    "Upload a clear, front-facing face photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Your Photo")
            st.image(image, width=350)

        with col2:
            st.subheader("AI Prediction")

            # Preprocess image
            img = image.resize((64, 64))
            img_array = np.array(img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)

            # Predict face shape
            predictions = model.predict(img_array, verbose=0)
            predicted_index = int(np.argmax(predictions[0]))

            if predicted_index >= len(class_names):
                st.error("The model output does not match the face-shape labels.")
                st.stop()

            face_shape = class_names[predicted_index]
            confidence = float(predictions[0][predicted_index]) * 100

            st.success(f"Predicted Face Shape: {face_shape}")
            st.write(f"Model confidence score: {confidence:.2f}%")

            # Recommendations
            result = recommendations.get(face_shape)

            if result:
                st.subheader("Recommended Frames")

                for frame in result["frames"]:
                    st.write(f"✓ {frame}")

                st.subheader("Recommended Colours")

                for colour in result["colours"]:
                    st.write(f"🎨 {colour}")

                st.caption(
                    "These are general style suggestions. "
                    "Actual suitability depends on fit and personal preference."
                )

    except Exception as e:
        st.error(f"Could not process this image: {e}")

st.divider()
st.caption("AI Glasses Recommender | Deep Learning Project")
