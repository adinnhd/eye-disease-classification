import numpy as np
from PIL import Image
import tensorflow as tf

def preprocess_image(image_file):
    """
    Preprocesses the image for MobileNetV2 inference.
    
    Args:
        image_file: The uploaded file object (from st.file_uploader).
        
    Returns:
        A preprocessed numpy array ready for model prediction.
    """
    # 1. Load image using PIL
    img = Image.open(image_file)
    
    # 2. Convert to RGB (ensure 3 channels)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # 3. Resize to 256x256 (as per training requirement)
    img = img.resize((256, 256))
    
    # 4. Convert to numpy array and float32
    img_array = np.array(img).astype(np.float32)
    
    # 5. Expand dims to create batch dimension (1, 256, 256, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 6. Returns raw [0, 255] because the model has internal Rescaling layers
    # Do NOT apply tf.keras.applications.mobilenet_v2.preprocess_input
    
    return img_array
