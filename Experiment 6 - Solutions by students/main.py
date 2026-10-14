import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
# STEP 1: Load image
image_path = "fruits.avif"
image = cv2.imread(image_path)
if image is None:
    print("Error: City.jpg not found!")
    exit()
print("Image loaded successfully.")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
os.makedirs("output", exist_ok=True)
plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.tight_layout()
plt.savefig(
    "output/01_original_image.png",
    dpi=150,
    bbox_inches="tight"
)
plt.show()
plt.close()
# STEP 2: Grayscale + Gaussian Blur
gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)
blurred = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)
print("Grayscale conversion completed.")
print("Gaussian Blur applied.")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(blurred, cmap="gray")
plt.title("Gaussian Blurred Image")
plt.axis("off")
plt.tight_layout()
plt.savefig(
    "output/02_grayscale_blurred.png",
    dpi=150,
    bbox_inches="tight"
)
plt.show()
plt.close()
# STEP 3: Global Thresholding
global_threshold_value = 127
ret_global, global_threshold = cv2.threshold(
    blurred,
    global_threshold_value,
    255,
    cv2.THRESH_BINARY
)
print("\nGlobal Thresholding")
print("-------------------")
print("Threshold value:", ret_global)
plt.figure(figsize=(10, 6))
plt.imshow(
    global_threshold,
    cmap="gray"
)
plt.title("Global Thresholding")
plt.axis("off")
plt.tight_layout()
plt.savefig(
    "output/03_global_threshold.png",
    dpi=150,
    bbox_inches="tight"
)
plt.show()
plt.close()
# STEP 4: Otsu's Thresholding
otsu_value, otsu_threshold = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
print("\nOtsu Thresholding")
print("-----------------")
print("Automatically calculated threshold:", otsu_value)
plt.figure(figsize=(10, 6))
plt.imshow(
    otsu_threshold,
    cmap="gray"
)
plt.title(
    "Otsu's Thresholding\nThreshold = "
    + str(round(otsu_value, 2))
)
plt.axis("off")
plt.tight_layout()
plt.savefig(
    "output/04_otsu_threshold.png",
    dpi=150,
    bbox_inches="tight"
)
plt.show()
plt.close()
# STEP 5: Adaptive Thresholding
adaptive_threshold = cv2.adaptiveThreshold(
    blurred,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)
print("\nAdaptive Thresholding")
print("---------------------")
print("Block size: 11")
print("Constant C: 2")
plt.figure(figsize=(10, 6))
plt.imshow(
    adaptive_threshold,
    cmap="gray"
)
plt.title("Adaptive Thresholding")
plt.axis("off")
plt.tight_layout()
plt.savefig(
    "output/05_adaptive_threshold.png",
    dpi=150,
    bbox_inches="tight"
)
plt.show()
plt.close()
# STEP 6: Watershed Segmentation
binary = otsu_threshold
kernel = np.ones(
    (3, 3),
    np.uint8
)
opening = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel,
    iterations=2
)
sure_background = cv2.dilate(
    opening,
    kernel,
    iterations=3
)
dist_transform = cv2.distanceTransform(
    opening,
    cv2.DIST_L2,
    5
)
ret, sure_foreground = cv2.threshold(
    dist_transform,
    0.5 * dist_transform.max(),
    255,
    0
)
sure_foreground = np.uint8(
    sure_foreground
)
unknown = cv2.subtract(
    sure_background,
    sure_foreground
)
ret, markers = cv2.connectedComponents(
    sure_foreground
)
markers = markers + 1
markers[unknown == 255] = 0
watershed_image = image.copy()
markers = cv2.watershed(
    watershed_image,
    markers
)
watershed_result = image_rgb.copy()
watershed_result[markers == -1] = [255, 0, 0]
print("\nWatershed Segmentation")
print("----------------------")
print("Watershed boundaries detected.")
plt.figure(figsize=(10, 6))
plt.imshow(
    watershed_result
)
plt.title("Watershed Segmentation")
plt.axis("off")
plt.tight_layout()
plt.savefig(
    "output/06_watershed.png",
    dpi=150,
    bbox_inches="tight"
)
plt.show()
plt.close()
# STEP 7: K-Means Color Segmentation
rgb_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)
pixel_values = rgb_image.reshape(
    (-1, 3)
)
pixel_values = np.float32(
    pixel_values
)
k = 4
print("\nK-Means Color Segmentation")
print("--------------------------")
print("Number of clusters:", k)
kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)
kmeans.fit(pixel_values)
centers = np.uint8(
    kmeans.cluster_centers_
)
labels = kmeans.labels_
segmented_pixels = centers[labels]
segmented_image = segmented_pixels.reshape(
    rgb_image.shape
)
plt.figure(figsize=(10, 6))
plt.imshow(
    segmented_image
)
plt.title(
    "K-Means Color Segmentation (K = 4)"
)
plt.axis("off")
plt.tight_layout()
plt.savefig(
    "output/07_kmeans_segmentation.png",
    dpi=150,
    bbox_inches="tight"
)
plt.show()
plt.close()
# STEP 8: Compare segmentation techniques
plt.figure(figsize=(16, 10))
plt.subplot(2, 3, 1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis("off")
plt.subplot(2, 3, 2)
plt.imshow(global_threshold, cmap="gray")
plt.title("Global Threshold")
plt.axis("off")
plt.subplot(2, 3, 3)
plt.imshow(otsu_threshold, cmap="gray")
plt.title("Otsu Threshold")
plt.axis("off")
plt.subplot(2, 3, 4)
plt.imshow(adaptive_threshold, cmap="gray")
plt.title("Adaptive Threshold")
plt.axis("off")
plt.subplot(2, 3, 5)
plt.imshow(watershed_result)
plt.title("Watershed")
plt.axis("off")
plt.subplot(2, 3, 6)
plt.imshow(segmented_image)
plt.title("K-Means")
plt.axis("off")
plt.tight_layout()
plt.savefig(
    "output/08_segmentation_comparison.png",
    dpi=150,
    bbox_inches="tight"
)
plt.show()
plt.close()
# STEP 9: Visualization and analysis
print("\n==============================")
print("SEGMENTATION ANALYSIS")
print("==============================")
print("""
Global Thresholding:
- Simple and fast.
- Works well when foreground and background
  have clearly different intensity values.
- Sensitive to illumination changes.
Otsu's Thresholding:
- Automatically calculates the threshold.
- Usually better than manually selecting a threshold.
- Works well when the image has two dominant intensity regions.
Adaptive Thresholding:
- Calculates a different threshold for local regions.
- Works well with uneven illumination.
- More computationally expensive than simple thresholding.
Watershed:
- Useful for separating touching or overlapping objects.
- Produces clear object boundaries.
- Requires good preprocessing and markers.
K-Means:
- Performs color-based segmentation.
- Groups pixels according to their color.
- Useful when different objects have different colors.
""")
print("\n==============================")
print("FINAL OBSERVATIONS")
print("==============================")
print("""
1. Global thresholding is simple and computationally efficient.
2. Otsu's method automatically chooses a suitable threshold
   and removes the need to manually select one.
3. Adaptive thresholding performs better when illumination
   is not uniform across the image.
4. Watershed is useful for separating touching objects and
   identifying object boundaries.
5. K-Means is useful for color-based segmentation and can
   divide an image into multiple color regions.
6. No single segmentation technique is best for every image.
   The technique should be selected according to the image
   characteristics and application.
Applications of Image Segmentation:
- Medical image analysis
- Tumor and organ detection
- Satellite image analysis
- Road and building detection
- Autonomous vehicles
- Object detection
- Face recognition
- Industrial inspection
- Agricultural image analysis
- Robotics
""")
# Save final output
print("\n======================================")
print("Experiment completed successfully!")
print("======================================")
print("\nAll output images are saved in:")
print("output/")
print("""
Generated files:
1. 01_original_image.png
2. 02_grayscale_blurred.png
3. 03_global_threshold.png
4. 04_otsu_threshold.png
5. 05_adaptive_threshold.png
6. 06_watershed.png
7. 07_kmeans_segmentation.png
8. 08_segmentation_comparison.png
""")