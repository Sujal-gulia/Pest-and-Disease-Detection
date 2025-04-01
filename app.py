import streamlit as st
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load model and processor
processor = AutoImageProcessor.from_pretrained("Diginsa/Plant-Disease-Detection-Project")
model = AutoModelForImageClassification.from_pretrained("Diginsa/Plant-Disease-Detection-Project")

st.title("Agricultural Pest Detector")
st.subheader("Upload crop images for pest identification")

uploaded_file = st.file_uploader("Choose field image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image")  # Updated deprecated parameter

    with col2:
        with st.spinner("Analyzing for pests..."):
            # Process image and generate predictions
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)

            st.subheader("Detection Results")

            # Extract class predictions
            predicted_class = outputs.logits.argmax(-1).item()

            # Attempt to retrieve class label if available
            id2label = model.config.id2label if hasattr(model.config, 'id2label') else {}
            predicted_label = id2label.get(predicted_class, f"Class {predicted_class}")

            st.write(f"Predicted category: {predicted_label}")