import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
import os
os.makedirs("output", exist_ok=True)
# STEP 1: Load images
image_path1 = "city1.avif"
image_path2 = "city2.jpg"
image1 = cv2.imread(image_path1)
image2 = cv2.imread(image_path2)
if image1 is None:
    print("Error: City.jpg not found!")
    exit()
if image2 is None:
    print("Error: City2.jpg not found!")
    exit()
print("Images loaded successfully.")
image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
# STEP 2: Grayscale and preprocessing
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
gray1_blur = cv2.GaussianBlur(gray1, (5, 5), 0)
gray2_blur = cv2.GaussianBlur(gray2, (5, 5), 0)
print("Grayscale conversion completed.")
cv2.imwrite("output/grayscale.jpg", gray1_blur)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(image1_rgb)
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(gray1_blur, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.tight_layout()
plt.savefig("output/01_original_grayscale.png", dpi=150)
plt.show()
plt.close()
# STEP 3: SIFT keypoints and descriptors
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(
    gray1_blur, None
)
keypoints2, descriptors2 = sift.detectAndCompute(
    gray2_blur, None
)
print("\nSIFT Feature Extraction")
print("-----------------------")
print("Keypoints in Image 1:", len(keypoints1))
print("Keypoints in Image 2:", len(keypoints2))
print("Descriptor shape Image 1:", descriptors1.shape)
print("Descriptor shape Image 2:", descriptors2.shape)
# STEP 4: Visualize SIFT keypoints
sift_image = cv2.drawKeypoints(
    image1_rgb,
    keypoints1,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
plt.figure(figsize=(12, 7))
plt.imshow(sift_image)
plt.title("SIFT Keypoints")
plt.axis("off")
plt.tight_layout()
plt.savefig(
    "output/02_sift_keypoints.png",
    dpi=150,
    bbox_inches="tight"
)
plt.show()
plt.close()
height, width = gray1.shape
regions = {
    "Top-Left": 0,
    "Top-Right": 0,
    "Bottom-Left": 0,
    "Bottom-Right": 0
}
for kp in keypoints1:
    x, y = kp.pt
    if x < width / 2 and y < height / 2:
        regions["Top-Left"] += 1
    elif x >= width / 2 and y < height / 2:
        regions["Top-Right"] += 1
    elif x < width / 2 and y >= height / 2:
        regions["Bottom-Left"] += 1
    else:
        regions["Bottom-Right"] += 1

print("\nSIFT Keypoint Distribution")
print("--------------------------")
for region, count in regions.items():
    print(region, ":", count)
plt.figure(figsize=(8, 5))
plt.bar(
    regions.keys(),
    regions.values()
)
plt.title("SIFT Keypoint Distribution")
plt.xlabel("Image Region")
plt.ylabel("Number of Keypoints")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(
    "output/03_keypoint_distribution.png",
    dpi=150
)
plt.show()
plt.close()
# STEP 5: HOG feature extraction
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
hog_features, hog_image = hog(
    gray1_blur,
    orientations=orientations,
    pixels_per_cell=pixels_per_cell,
    cells_per_block=cells_per_block,
    block_norm="L2-Hys",
    visualize=True
)
print("\nHOG Feature Extraction")
print("----------------------")
print("Number of HOG features:", len(hog_features))
# STEP 6: HOG visualization
hog_image_rescaled = exposure.rescale_intensity(
    hog_image,
    in_range=(0, 10)
)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray1_blur, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(hog_image_rescaled, cmap="gray")
plt.title("HOG Descriptor")
plt.axis("off")
plt.tight_layout()
plt.savefig(
    "output/04_hog_descriptor.png",
    dpi=150,
    bbox_inches="tight"
)
plt.show()
plt.close()
# STEP 7: SIFT vs HOG comparison
print("\nSIFT vs HOG")
print("-----------")
print("""
SIFT:
- Detects local keypoints.
- Scale invariant.
- Rotation invariant.
- Good for image matching.
- More computationally expensive.
HOG:
- Describes edge and gradient information.
- Not inherently scale invariant.
- Not inherently rotation invariant.
- Good for object and shape detection.
- Generally computationally efficient.
""")
# STEP 8: SIFT Image Matching
index_params = dict(
    algorithm=1,
    trees=5
)
search_params = dict(
    checks=50
)
flann = cv2.FlannBasedMatcher(
    index_params,
    search_params
)
matches = flann.knnMatch(
    descriptors1,
    descriptors2,
    k=2
)
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
print("\nImage Matching")
print("--------------")
print("Total matches:", len(matches))
print("Good matches:", len(good_matches))
matched_image = cv2.drawMatches(
    image1_rgb,
    keypoints1,
    image2_rgb,
    keypoints2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
plt.figure(figsize=(16, 8))
plt.imshow(matched_image)
plt.title("SIFT Image Matching")
plt.axis("off")
plt.tight_layout()
plt.savefig(
    "output/05_sift_matching.png",
    dpi=150,
    bbox_inches="tight"
)
plt.show()
plt.close()
# HOG comparison between two images
hog_features2, hog_image2 = hog(
    gray2_blur,
    orientations=orientations,
    pixels_per_cell=pixels_per_cell,
    cells_per_block=cells_per_block,
    block_norm="L2-Hys",
    visualize=True
)
if len(hog_features) == len(hog_features2):
    dot_product = np.dot(
        hog_features,
        hog_features2
    )
    norm1 = np.linalg.norm(hog_features)
    norm2 = np.linalg.norm(hog_features2)
    hog_similarity = dot_product / (
        norm1 * norm2
    )
    print("\nHOG Similarity:", hog_similarity)
    if hog_similarity > 0.8:
        print("Result: Images are highly similar.")
    elif hog_similarity > 0.5:
        print("Result: Images have moderate similarity.")
    else:
        print("Result: Images are not very similar.")
# STEP 9: Strengths and limitations
print("\nStrengths and Limitations")
print("-------------------------")
print("""
SIFT Strengths:
1. Scale invariant
2. Rotation invariant
3. Robust to image transformations
4. Excellent for image matching
SIFT Limitations:
1. Computationally expensive
2. Requires more memory
3. Can produce many descriptors
HOG Strengths:
1. Good for shape detection
2. Good for object detection
3. Efficient for fixed-size images
4. Works well with SVM classifiers
HOG Limitations:
1. Not scale invariant
2. Not rotation invariant
3. Sensitive to large orientation changes
4. Requires suitable image size
""")
# STEP 10: Applications
print("\nApplications")
print("------------")
print("""
SIFT:
- Image matching
- Object recognition
- Panorama stitching
- Image registration
- Feature tracking
HOG:
- Pedestrian detection
- Human detection
- Object detection
- Shape recognition
- Image classification
""")
# Final output
print("\n======================================")
print("Experiment completed successfully!")
print("======================================")
print("\nOutput images have been saved in:")
print("output/")
print("""
Generated files:
1. 01_original_grayscale.png
2. 02_sift_keypoints.png
3. 03_keypoint_distribution.png
4. 04_hog_descriptor.png
5. 05_sift_matching.png
""")