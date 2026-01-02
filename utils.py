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
    
    # 6. Apply MobileNetV2 preprocessing (scales pixel values to [-1, 1])
    # Standard MobileNetV2 expects [-1, 1]
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    return img_array

def preprocess_image_debug(image_file, mode="mobilenet"):
    """
    Debug version of preprocessing to test different scaling methods.
    """
    img = Image.open(image_file).convert("RGB").resize((256, 256))
    img_array = np.array(img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    if mode == "mobilenet":
        # [-1, 1]
        return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    elif mode == "rescaling":
        # [0, 1]
        return img_array / 255.0
    else:
        # [0, 255] (Raw)
        return img_array
