import onnxruntime as ort
import cv2
import numpy as np
import os
from glob import glob

# ===== Configuration =====
model_name = 'AnimeGANv3_Hayao_STYLE_36'
in_dir = 'input'
out_dir = 'output'
max_dimension = 1280  # Keeps quality while preventing memory issues

# Initialize ONNX Runtime
session = ort.InferenceSession(f'{model_name}.onnx', providers=['CPUExecutionProvider'])

def make_dimensions_valid(img):
    """Ensure image dimensions are divisible by 8"""
    h, w = img.shape[:2]
    new_h = h if h % 8 == 0 else ((h // 8) + 1) * 8
    new_w = w if w % 8 == 0 else ((w // 8) + 1) * 8
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load {img_path}")
    
    # Resize if too large (maintains aspect ratio)
    h, w = img.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    # Ensure valid dimensions for ONNX
    img = make_dimensions_valid(img)
    
    # Normalize for model
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    return np.expand_dims(img, axis=0), (h, w)  # Return original dimensions for resizing back

def convert_to_painting(img_tensor, original_size):
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img_tensor})[0]
    
    # Post-process
    output = (np.squeeze(output) + 1.0) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    
    # Resize back to original dimensions
    return cv2.resize(output, (original_size[1], original_size[0]), interpolation=cv2.INTER_CUBIC)

def process_directory():
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    for img_path in glob(os.path.join(in_dir, '*')):
        try:
            print(f"Processing {os.path.basename(img_path)}...")
            img_tensor, original_size = process_image(img_path)
            result = convert_to_painting(img_tensor, original_size)
            
            out_path = os.path.join(out_dir, f"painting_{os.path.basename(img_path)}")
            cv2.imwrite(out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"Saved: {out_path}")
        
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

if __name__ == "__main__":
    print("===== Painting Converter =====")
    process_directory()