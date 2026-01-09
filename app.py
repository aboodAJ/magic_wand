import streamlit as st
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import io

# Import project modules
# Assuming src is in the python path or same directory. 
# If running from root, these imports should work if __init__.py exists or python path is set.
# We will use sys.path to be safe.
import sys
import os
sys.path.append(os.path.abspath('.'))

from src.model import UNet
from src.utils import denormalize

# --- Page Config & Styling ---
st.set_page_config(
    page_title="Magic Wand Segmentation",
    page_icon="🪄",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #0e1117;
        color: #f0f2f6;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        background: linear-gradient(90deg, #A855F7, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    /* Cards/Containers */
    .css-1r6slb0, .css-12oz5g7 {
        background-color: #1f2937;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #374151;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
    }
    
    /* Image Containers */
    .img-container {
        border-radius: 12px;
        overflow: hidden;
        border: 2px solid #374151;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Configuration ---
MODEL_PATH = 'unet_coco_epoch_5.pth'
IMG_SIZE = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Helper Functions ---

@st.cache_resource
def load_model():
    """Loads the UNet model (cached)."""
    model = UNet(n_channels=3, n_classes=1) 
    try:
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
        else:
            st.error(f"Model file not found at {MODEL_PATH}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
        
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(image_np):
    """Resizes and normalizes image for the model."""
    transform = A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    augmented = transform(image=image_np)
    return augmented['image'].unsqueeze(0).to(DEVICE) # (1, 3, H, W)

def get_blob_mask(prob_map, click_scaled, threshold=0.5):
    """
    Extracts the specific blob selected by the user.
    prob_map: (512, 512) float array 0..1
    click_scaled: (x, y) tuple in 512x512 space
    threshold: float, dynamic threshold for binarization
    """
    # Threshold
    binary_mask = (prob_map > threshold).astype(np.uint8)
    
    # Connected Components
    # connectivity=8 implies 8-way connectivity (pixels connected horizontally, vertically, or diagonally)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # Identify clicked label
    # Note: labels is [y, x], click_scaled is (x, y)
    cx, cy = int(click_scaled[0]), int(click_scaled[1])
    
    # Boundary check
    cx = np.clip(cx, 0, IMG_SIZE - 1)
    cy = np.clip(cy, 0, IMG_SIZE - 1)
    
    selected_label = labels[cy, cx]
    
    if selected_label == 0:
        return None  # Background clicked
        
    blob_mask = (labels == selected_label).astype(np.uint8)
    return blob_mask

def refine_edges(original_image_np, blob_mask_full_size):
    """
    Refines the rough blob mask using GrabCut.
    original_image_np: (H, W, 3) BGR or RGB
    blob_mask_full_size: (H, W) binary mask (0 or 1)
    """
    # GrabCut Logic
    # Mask values: 0: GC_BGD, 1: GC_FGD, 2: GC_PR_BGD, 3: GC_PR_FGD
    
    # Initialize mask for GrabCut
    # Everything is Sure BG (0) by default
    gc_mask = np.zeros(blob_mask_full_size.shape, dtype=np.uint8)
    
    # Set the blob area to Probable FG (3)
    gc_mask[blob_mask_full_size == 1] = cv2.GC_PR_FGD
    
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # Run GrabCut initialized with mask
    # Iterations=5 is usually sufficient
    try:
        cv2.grabCut(original_image_np, gc_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    except Exception as e:
        st.warning(f"GrabCut failed: {e}")
        return blob_mask_full_size

    # Isolate foreground (Sure FG (1) + Prob FG (3))
    final_mask = np.where((gc_mask == 2) | (gc_mask == 0), 0, 1).astype(np.uint8)
    return final_mask

# --- Main App ---

def main():
    st.title("🪄 Magic Wand Segmentation")
    st.write("Upload an image, adjust the heatmap threshold, and click to extract!")

    # Sidebar parameters
    with st.sidebar:
        st.header("Settings")
        threshold = st.slider("Segmentation Threshold", 0.0, 1.0, 0.5, 0.01, help="Adjust sensitivity of the blob detection.")
        alpha = st.slider("Heatmap Opacity", 0.0, 1.0, 0.4, 0.1)
        st.info("Model: U-Net + GrabCut \n Dataset: COCO")

    # 1. File Upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load and convert image
        image = Image.open(uploaded_file).convert('RGB')
        original_w, original_h = image.size
        
        image_np_rgb = np.array(image)
        image_np_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR)

        # 2. Auto-Inference (Run as soon as image is loaded)
        model = load_model()
        if model is None:
            return

        with st.spinner("Analyzing image..."):
            input_tensor = preprocess_image(image_np_rgb)
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.sigmoid(logits)
                prob_map = probs.squeeze().cpu().numpy() # (512, 512)

        # 3. Generate Heatmap Overlay
        # Resize probability map to original image size for display
        prob_map_full = cv2.resize(prob_map, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        # Colorize
        heatmap_uint8 = (prob_map_full * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) # RGB for PIL
        
        # Blend
        blended = cv2.addWeighted(image_np_rgb, 1 - alpha, heatmap_color, alpha, 0)
        blended_pil = Image.fromarray(blended)

        # 4. Display & Interaction
        st.write("### Click on the heatmap to select an object:")
        
        with st.container():
             value = streamlit_image_coordinates(
                blended_pil,
                key="pil",
            )

        if value:
            click_x = value['x']
            click_y = value['y']
            
            with st.spinner("Refining selection..."):
                # --- Step A: Blob Selection ---
                # Map click coords to 512x512 (Model Space)
                scale_x = IMG_SIZE / original_w
                scale_y = IMG_SIZE / original_h
                
                scaled_click_x = click_x * scale_x
                scaled_click_y = click_y * scale_y
                
                # Use the dynamic threshold from sidebar
                blob_mask_512 = get_blob_mask(prob_map, (scaled_click_x, scaled_click_y), threshold=threshold)
                
                if blob_mask_512 is None:
                    st.error(f"No object detected at ({click_x}, {click_y}) with threshold {threshold}. Try lowering the sensitivity or clicking a clearer area.")
                else:
                    # --- Step B: Refinement ---
                    # Upscale blob mask to original size
                    blob_mask_full = cv2.resize(blob_mask_512, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
                    
                    # Run GrabCut
                    final_mask = refine_edges(image_np_bgr, blob_mask_full)
                    
                    # --- Visualization ---
                    
                    # Create Cutout
                    cutout = image_np_rgb.copy()
                    cutout[final_mask == 0] = 0
                    
                    # Create transparent version
                    cutout_rgba = cv2.cvtColor(cutout, cv2.COLOR_RGB2RGBA)
                    cutout_rgba[:, :, 3] = final_mask * 255
                    
                    st.success("Segmentation Complete!")
                    
                    cols = st.columns(3)
                    
                    with cols[0]:
                        st.subheader("Blended Input")
                        st.image(blended_pil, use_container_width=True)
                        
                    with cols[1]:
                        st.subheader("Selected Blob")
                        # Show the specific blob mask
                        st.image(blob_mask_full * 255, clamp=True, channels='GRAY', use_container_width=True, caption=f"Threshold: {threshold}")
                        
                    with cols[2]:
                        st.subheader("GrabCut Result")
                        st.image(cutout, use_container_width=True)
                        
                    # Download Button
                    cutout_pil = Image.fromarray(cutout_rgba)
                    buf = io.BytesIO()
                    cutout_pil.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="Download Cutout (PNG)",
                        data=byte_im,
                        file_name="cutout.png",
                        mime="image/png"
                    )

if __name__ == "__main__":
    main()
