# STEP 1: Import required libraries and load an image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
# Path to the input image. Replace with your own image file.
# If no image is found, a synthetic test image with noise is generated
# so the script still runs end-to-end.
IMAGE_PATH = r"C:\Users\DELL\Downloads\Animal.jpg"
if os.path.exists(IMAGE_PATH):
    img_color = cv2.imread(IMAGE_PATH)              # BGR color image
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
else:
    print(f"'{IMAGE_PATH}' not found — generating a synthetic test image instead.")
    # Create a synthetic grayscale image with shapes + noise
    img_gray = np.full((300, 300), 200, dtype=np.uint8)
    cv2.rectangle(img_gray, (50, 50), (150, 150), 50, -1)
    cv2.circle(img_gray, (200, 200), 60, 120, -1)
    cv2.line(img_gray, (0, 0), (299, 299), 0, 2)
    # Add Gaussian noise
    noise = np.random.normal(0, 15, img_gray.shape).astype(np.int16)
    img_gray = np.clip(img_gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Add salt-and-pepper noise
    sp_mask = np.random.random(img_gray.shape)
    img_gray[sp_mask < 0.02] = 0
    img_gray[sp_mask > 0.98] = 255
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
# ---------------------------------------------------------------------------
# STEP 2: Display the original image and analyze its visual characteristics
# ---------------------------------------------------------------------------
print("Image shape:", img_gray.shape)
print("Mean intensity:", round(float(np.mean(img_gray)), 2))
print("Std deviation (contrast):", round(float(np.std(img_gray)), 2))
plt.figure(figsize=(5, 5))
plt.imshow(img_gray, cmap="gray")
plt.title("Original Grayscale Image")
plt.axis("off")
plt.show()
# ---------------------------------------------------------------------------
# STEP 3: Gaussian Blur — low-pass filter for noise reduction
# ---------------------------------------------------------------------------
t0 = time.time()
gaussian_blur = cv2.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=1.0)
t_gaussian = time.time() - t0
# ---------------------------------------------------------------------------
# STEP 4: Median Filtering — removes salt-and-pepper (impulse) noise
# ---------------------------------------------------------------------------
t0 = time.time()
median_filtered = cv2.medianBlur(img_gray, ksize=5)
t_median = time.time() - t0
# ---------------------------------------------------------------------------
# STEP 5: Average (Mean) Filtering — compare smoothing vs Gaussian
# ---------------------------------------------------------------------------
t0 = time.time()
average_filtered = cv2.blur(img_gray, ksize=(5, 5))
t_average = time.time() - t0
# ---------------------------------------------------------------------------
# STEP 6: Laplacian Filtering — high-pass filter, enhances edges/details
# ---------------------------------------------------------------------------
t0 = time.time()
laplacian = cv2.Laplacian(img_gray, ddepth=cv2.CV_64F, ksize=3)
laplacian_abs = cv2.convertScaleAbs(laplacian)   # convert back to uint8 for display
t_laplacian = time.time() - t0
# Optional: Laplacian sharpened image (original - laplacian highlights edges)
sharpened = cv2.convertScaleAbs(img_gray.astype(np.float64) - laplacian)
# ---------------------------------------------------------------------------
# STEP 7: Sobel Edge Detection — horizontal and vertical gradients
# ---------------------------------------------------------------------------
t0 = time.time()
sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, dx=1, dy=0, ksize=3)   # horizontal gradient
sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, dx=0, dy=1, ksize=3)   # vertical gradient
sobel_combined = cv2.magnitude(sobel_x, sobel_y)                  # combined gradient magnitude
t_sobel = time.time() - t0
sobel_x_abs = cv2.convertScaleAbs(sobel_x)
sobel_y_abs = cv2.convertScaleAbs(sobel_y)
sobel_combined_abs = cv2.convertScaleAbs(sobel_combined)
# ---------------------------------------------------------------------------
# STEP 8: Compare filters — timing + simple quality metrics
# ---------------------------------------------------------------------------
def edge_strength(image):
    """Mean absolute Laplacian value — a proxy for edge/detail strength."""
    lap = cv2.Laplacian(image, cv2.CV_64F)
    return float(np.mean(np.abs(lap)))
def noise_estimate(image):
    """Std deviation of Laplacian — rough proxy for noise/graininess."""
    lap = cv2.Laplacian(image, cv2.CV_64F)
    return float(np.std(lap))
print("\n--- Filter Comparison ---")
print(f"{'Filter':<20}{'Time (s)':<12}{'Edge Strength':<16}{'Noise Estimate'}")
for name, out, t in [
    ("Original", img_gray, 0.0),
    ("Gaussian Blur", gaussian_blur, t_gaussian),
    ("Median Filter", median_filtered, t_median),
    ("Average Filter", average_filtered, t_average),
]:
    print(f"{name:<20}{t:<12.5f}{edge_strength(out):<16.2f}{noise_estimate(out):.2f}")

print(f"\nLaplacian computation time: {t_laplacian:.5f} s")
print(f"Sobel computation time:     {t_sobel:.5f} s")
# ---------------------------------------------------------------------------
# STEP 9: Display original and all filtered images together
# ---------------------------------------------------------------------------
images = [
    ("Original", img_gray),
    ("Gaussian Blur (Low-Pass)", gaussian_blur),
    ("Median Filter (Low-Pass)", median_filtered),
    ("Average Filter (Low-Pass)", average_filtered),
    ("Laplacian (High-Pass)", laplacian_abs),
    ("Laplacian Sharpened", sharpened),
    ("Sobel X (Horizontal)", sobel_x_abs),
    ("Sobel Y (Vertical)", sobel_y_abs),
    ("Sobel Combined", sobel_combined_abs),
]
plt.figure(figsize=(15, 12))
for i, (title, im) in enumerate(images, 1):
    plt.subplot(3, 3, i)
    plt.imshow(im, cmap="gray")
    plt.title(title, fontsize=10)
    plt.axis("off")
plt.tight_layout()
plt.savefig("filter_comparison.png", dpi=150)
plt.show()
print("\nComparison figure saved as 'filter_comparison.png'.")
# ---------------------------------------------------------------------------
# STEP 10: Interpretation / Observations
# ---------------------------------------------------------------------------
print("""
--- Observations ---
1. Gaussian Blur smooths the image while giving more weight to central pixels,
   preserving edges better than a plain average filter but still reducing
   fine detail as kernel size increases.
2. Median Filtering is highly effective against salt-and-pepper noise because
   it replaces each pixel with the median of its neighborhood, which is
   robust to extreme outlier values (unlike the mean).
3. Average (Mean) Filtering blurs uniformly and tends to blur edges more than
   Gaussian blur for the same kernel size, since all neighboring pixels
   contribute equally.
4. Laplacian Filtering is highly sensitive to noise (it is a second-derivative
   operator) but is excellent at highlighting fine edges and details; it is
   often applied after a smoothing step in practice.
5. Sobel Edge Detection highlights directional gradients — Sobel X detects
   vertical edges (horizontal intensity change) and Sobel Y detects
   horizontal edges (vertical intensity change); combining them gives the
   overall edge map.
6. There is a clear trade-off: low-pass filters reduce noise but blur edges,
   while high-pass filters sharpen/detect edges but amplify noise.
""")