import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim  # Make sure scikit-image is updated

# Generate a synthetic 2D array (grayscale image)
image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)  # Random grayscale image

# 1. **Brightness and Contrast Adjustment**
def adjust_brightness_contrast(img, alpha=1.5, beta=50):
    """
    Adjusts the brightness and contrast of the image.
    - alpha: Contrast control (1.0-3.0)
    - beta: Brightness control (0-100)
    """
    return np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

# 2. **Histogram Equalization** (Improves contrast in an image)
def histogram_equalization(img):
    # Compute the histogram and normalize it
    img_hist_eq = np.zeros_like(img)
    unique, counts = np.unique(img, return_counts=True)
    cdf = np.cumsum(counts) / np.sum(counts)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_hist_eq[i, j] = np.round(cdf[img[i, j]] * 255)
    return img_hist_eq.astype(np.uint8)

# 3. **Gamma Correction** (Non-linear image enhancement)
def gamma_correction(img, gamma=1.5):
    invGamma = 1.0 / gamma
    lookup_table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)], dtype="uint8")
    return lookup_table[img].astype(np.uint8)

# 4. **Image Sharpening** (Enhance edges and details)
def sharpen_image(img):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])  # Simple sharpening kernel
    img_sharpened = np.zeros_like(img)
    # Apply the kernel (convolution operation)
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            img_sharpened[i, j] = np.sum(kernel * img[i-1:i+2, j-1:j+2])
    img_sharpened = np.clip(img_sharpened, 0, 255)
    return img_sharpened.astype(np.uint8)

# Apply image enhancement techniques
bright_contrast_img = adjust_brightness_contrast(image, alpha=1.8, beta=30)
hist_eq_img = histogram_equalization(image)
gamma_corrected_img = gamma_correction(image, gamma=1.5)
sharpened_img = sharpen_image(image)

# Convert images to float32 for SSIM computation
# We convert to float and normalize them to [0, 1] range for SSIM
image_float = image.astype(np.float32) / 255.0
bright_contrast_img_float = bright_contrast_img.astype(np.float32) / 255.0
hist_eq_img_float = hist_eq_img.astype(np.float32) / 255.0
gamma_corrected_img_float = gamma_corrected_img.astype(np.float32) / 255.0
sharpened_img_float = sharpened_img.astype(np.float32) / 255.0

# Compute SSIM for each enhanced image with data_range set to 1.0 (since images are in the range [0, 1])
gamma_ssim = ssim(image_float, gamma_corrected_img_float, data_range=1.0)
hist_eq_ssim = ssim(image_float, hist_eq_img_float, data_range=1.0)
bright_contrast_ssim = ssim(image_float, bright_contrast_img_float, data_range=1.0)
sharpened_ssim = ssim(image_float, sharpened_img_float, data_range=1.0)

# Plotting the original and enhanced images
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Original Image
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title("Original Image")
axes[0, 0].axis('off')

# Brightness and Contrast Adjusted Image
axes[0, 1].imshow(bright_contrast_img, cmap='gray')
axes[0, 1].set_title("Brightness & Contrast")
axes[0, 1].axis('off')

# Histogram Equalized Image
axes[0, 2].imshow(hist_eq_img, cmap='gray')
axes[0, 2].set_title("Histogram Equalization")
axes[0, 2].axis('off')

# Gamma Corrected Image
axes[1, 0].imshow(gamma_corrected_img, cmap='gray')
axes[1, 0].set_title("Gamma Correction")
axes[1, 0].axis('off')

# Sharpened Image
axes[1, 1].imshow(sharpened_img, cmap='gray')
axes[1, 1].set_title("Sharpened Image")
axes[1, 1].axis('off')

# Display SSIM values
axes[1, 2].text(0.1, 0.8, f"Gamma Correction: \n SSIM: {gamma_ssim:.4f}", fontsize=12)
axes[1, 2].text(0.1, 0.6, f"Histogram Equalization: \n SSIM: {hist_eq_ssim:.4f}", fontsize=12)
axes[1, 2].text(0.1, 0.4, f"Brightness & Contrast: \n SSIM: {bright_contrast_ssim:.4f}", fontsize=12)
axes[1, 2].text(0.1, 0.2, f"Sharpened Image: \n SSIM: {sharpened_ssim:.4f}", fontsize=12)

axes[1, 2].axis('off')

# Show the images
plt.tight_layout()
plt.show()
