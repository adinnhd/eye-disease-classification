import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from utils import preprocess_image

# 1. Page Configuration
st.set_page_config(
    page_title="Eye Disease Classifier",
    page_icon="👁️",
    layout="centered"
)

# 2. Constants & Paths
MODEL_PATH = "saved_models/mobilenet_fundus.keras"
CLASS_NAMES_PATH = "saved_models/class_names.json"

# 3. Load Resources (Cached)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: `{MODEL_PATH}`")
        st.info("Please make sure you have uploaded the model file to the correct path in your repository.")
        st.stop()
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

@st.cache_data
def load_class_names():
    if not os.path.exists(CLASS_NAMES_PATH):
        st.error(f"❌ Class names file not found at: `{CLASS_NAMES_PATH}`")
        st.info("Please create a `class_names.json` file in `saved_models/` containing a list of your class names.")
        st.stop()
        
    try:
        with open(CLASS_NAMES_PATH, "r") as f:
            class_names = json.load(f)
        return class_names
    except Exception as e:
        st.error(f"❌ Error loading class names: {e}")
        st.stop()

# 4. Main App UI
def main():
    st.title("👁️ Eye Disease Classification")
    st.write("Upload a fundus image (retina) to detect potential eye diseases.")
    
    # Load model and classes
    model = load_model()
    class_names = load_class_names()
    
    # File Uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Sidebar for Debugging
    st.sidebar.title("🔧 Debug Options")
    use_debug_mode = st.sidebar.checkbox("Activating Debug Mode", value=False)
    
    preprocess_mode = "mobilenet"
    if use_debug_mode:
        preprocess_mode = st.sidebar.selectbox(
            "Preprocessing Mode",
            ["mobilenet", "rescaling", "raw"],
            help="mobilenet: [-1, 1], rescaling: [0, 1], raw: [0, 255]"
        )

    if uploaded_file is not None:
        # Display Image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        # Predict Button
        if st.button("🔍 Predict Disease"):
            with st.spinner("Analyzing image..."):
                try:
                    # Preprocess
                    if use_debug_mode:
                        from utils import preprocess_image_debug
                        input_tensor = preprocess_image_debug(uploaded_file, mode=preprocess_mode)
                        
                        # Show stats
                        st.sidebar.write("Input Stats:")
                        st.sidebar.write(f"Min: {input_tensor.min():.2f}")
                        st.sidebar.write(f"Max: {input_tensor.max():.2f}")
                        st.sidebar.write(f"Mean: {input_tensor.mean():.2f}")
                        st.sidebar.write(f"Shape: {input_tensor.shape}")
                    else:
                        input_tensor = preprocess_image(uploaded_file)
                    
                    # Inference
                    logits = model.predict(input_tensor)
                    
                    # Post-processing
                    probabilities = tf.nn.softmax(logits).numpy()[0]
                    predicted_index = np.argmax(probabilities)
                    predicted_class = class_names[predicted_index]
                    confidence = probabilities[predicted_index] * 100
                    
                    if use_debug_mode:
                        st.subheader("🛠️ Deep Debugging")
                        
                        # 1. Input Stats
                        st.write("**Input Tensor Stats:**")
                        st.json({
                            "min": float(input_tensor.min()),
                            "max": float(input_tensor.max()),
                            "mean": float(input_tensor.mean()),
                            "shape": str(input_tensor.shape)
                        })
                        
                        # 2. Raw Logits
                        st.write("**Raw Logits (Pre-Softmax):**")
                        st.write(logits[0].tolist())

                        # 3. Model Architecture
                        with st.expander("See Model Summary"):
                            stringlist = []
                            model.summary(print_fn=lambda x: stringlist.append(x))
                            short_summary = "\n".join(stringlist)
                            st.code(short_summary)
                    
                    # Display Results
                    st.success("Analysis Complete!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="Predicted Class", value=predicted_class)
                    with col2:
                        st.metric(label="Confidence", value=f"{confidence:.2f}%")
                    
                    # Probability Chart
                    st.subheader("Probability Distribution")
                    probs_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
                    st.bar_chart(probs_dict)

if __name__ == "__main__":
    main()
