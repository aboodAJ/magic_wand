import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Denormalizes a tensor image to be visualized."""
    # Clone to avoid modifying original
    tensor = tensor.clone().detach().cpu()
    
    # Check if batch dimension exists
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
        
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    tensor = tensor * std + mean
    return tensor.clamp(0, 1)

def tensor_to_numpy(tensor):
    """Converts a tensor image (CHW) to numpy (HWC)."""
    img = denormalize(tensor)
    return img.permute(1, 2, 0).numpy()

def visualize(image=None, mask=None, prediction=None, grabcut_mask=None):
    """Plotting utility for side-by-side comparison."""
    n_images = sum(x is not None for x in [image, mask, prediction, grabcut_mask])
    if n_images == 0:
        return

    plt.figure(figsize=(5 * n_images, 5))
    i = 1
    
    if image is not None:
        plt.subplot(1, n_images, i)
        plt.title('Image')
        if torch.is_tensor(image):
            plt.imshow(tensor_to_numpy(image))
        else:
            plt.imshow(image)
        plt.axis('off')
        i += 1
        
    if mask is not None:
        plt.subplot(1, n_images, i)
        plt.title('Ground Truth Mask')
        if torch.is_tensor(mask):
            plt.imshow(mask.squeeze().cpu().numpy(), cmap='gray')
        else:
            plt.imshow(mask, cmap='gray')
        plt.axis('off')
        i += 1
        
    if prediction is not None:
        plt.subplot(1, n_images, i)
        plt.title('U-Net Prediction')
        if torch.is_tensor(prediction):
            pred_map = prediction.squeeze().cpu().numpy()
        else:
            pred_map = prediction
        plt.imshow(pred_map, cmap='jet', vmin=0, vmax=1)
        plt.colorbar()
        plt.axis('off')
        i += 1

    if grabcut_mask is not None:
        plt.subplot(1, n_images, i)
        plt.title('GrabCut Mask')
        # Map values for better visualization: 
        # 0:Definite BG, 1:Definite FG, 2:Probable BG, 3:Probable FG
        plt.imshow(grabcut_mask, vmin=0, vmax=3, cmap='viridis') 
        plt.axis('off')
        i += 1
        
    plt.tight_layout()
    plt.show()

def prob_to_grabcut_mask(prob_map, low_thresh=0.1, high_thresh=0.9):
    """
    Converts a probability map (0-1) to a GrabCut mask (0, 1, 2, 3).
    
    cv2.GC_BGD = 0  (Sure Background)
    cv2.GC_FGD = 1  (Sure Foreground)
    cv2.GC_PR_BGD = 2 (Probable Background)
    cv2.GC_PR_FGD = 3 (Probable Foreground)
    """
    mask = np.full(prob_map.shape, cv2.GC_PR_FGD, dtype=np.uint8) # Default to Probable FG
    
    mask[prob_map < low_thresh] = cv2.GC_BGD       # Sure BG
    mask[prob_map > high_thresh] = cv2.GC_FGD      # Sure FG
    
    return mask
