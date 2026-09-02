# Coral Colony Health Analysis

## MATE ROV 2021 – Task 2.2 Style

### Introduction and Apology

First, I would like to apologize for the late submission of this assignment. I needed additional time to understand the problem, develop the solution, test the code, and make sure that the analysis pipeline was working correctly.

For this implementation, I chose to test my solution using **two random images from the available dataset** rather than processing the entire database. The purpose of this initial implementation was to build and test the complete computer vision pipeline on one image pair first. Once the pipeline is verified and working properly, the same approach can be extended to automatically process multiple image pairs from the full dataset.

---

# 1. Problem Description

The goal of this project is to analyze the health of a coral colony by comparing two images of the same coral taken at different times.

The system compares a **Before image** and an **After image** to detect changes in the coral colony.

The code is designed to detect four main types of changes:

* **Growth:** New coral material appears in the After image.
* **Damage or Death:** Coral material that existed in the Before image is missing in the After image.
* **Bleaching:** Healthy coral changes from pink to white.
* **Recovery:** Previously bleached coral changes from white back to pink.

For this practice setup:

* **Pink coral represents healthy coral.**
* **White coral represents bleached coral.**

---

# 2. Solution Approach

The solution was implemented using Python and Computer Vision techniques.

The main libraries used were:

* OpenCV
* NumPy
* Matplotlib
* Google Colab file upload tools

The complete solution follows the following steps:

1. Upload the images.
2. Load and resize the images.
3. Align the After image with the Before image.
4. Segment the coral from the background.
5. Classify coral pixels as healthy or bleached.
6. Compare the two images.
7. Detect growth, damage, bleaching, and recovery.
8. Remove small noise regions.
9. Create a visual overlay.
10. Calculate statistics.
11. Display and save the final result.

---

# 3. Image Upload

The code was implemented and tested using Google Colab.

The following code allows the user to upload images directly from their computer:

```python
from google.colab import files

uploaded = files.upload()
```

The user uploads two images:

* Before image
* After image

For this assignment, I selected two random images from the available dataset to test the solution.

The selected images were:

* `coral3.jpg`
* `coral4.jpg`

These images were used only for testing the implementation.

---

# 4. Image Loading and Resizing

The images may have different resolutions. Therefore, both images are resized to the same standard size before processing.

```python
STD_SIZE = (900, 900)
```

The function `load_image()` reads the image using OpenCV and resizes it.

```python
img = cv2.imread(path)
img = cv2.resize(img, STD_SIZE)
```

Using the same image size makes pixel comparison easier and ensures that the measurements are consistent.

---

# 5. Image Alignment

One of the main challenges is that the Before and After images may not be taken from exactly the same position.

The camera may have differences in:

* Position
* Angle
* Rotation
* Distance
* Perspective

Therefore, the images need to be aligned before comparing them.

The code uses:

* ORB feature detection
* Feature matching
* RANSAC
* Homography
* Perspective transformation

## ORB Feature Detection

ORB detects important feature points in both images.

These features may include:

* Corners
* Pipe joints
* Edges
* Other unique visual patterns

The code detects these features using:

```python
orb = cv2.ORB_create(max_features)
```

---

## Feature Matching

After detecting features, the code matches similar features between the two images.

```python
matcher = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=True
)
```

The matches are sorted based on their quality.

The best matches are selected and used for alignment.

---

## Homography

A homography matrix is calculated using the matched feature points.

```python
homography, mask = cv2.findHomography(
    pts1,
    pts2,
    cv2.RANSAC,
    5.0
)
```

RANSAC helps remove incorrect feature matches.

The homography matrix is then used to transform the After image so that it aligns with the Before image.

```python
aligned = cv2.warpPerspective(
    img_to_warp,
    homography,
    (w, h)
)
```

After this step, both images are approximately in the same coordinate system.

---

# 6. Coral Segmentation

The next step is to separate the coral from the background.

The images contain:

* Pink coral pipes
* White coral pipes
* Background cloth

The code converts the image from BGR color space to HSV color space.

```python
hsv = cv2.cvtColor(
    img_bgr,
    cv2.COLOR_BGR2HSV
)
```

HSV was chosen because it separates:

* Hue
* Saturation
* Value

This makes color-based detection easier and more robust.

---

# 7. Healthy Coral Detection

Healthy coral is represented by the color pink.

The code detects pink pixels using HSV thresholds.

```python
PINK_HUE_MIN = 105
PINK_HUE_MAX = 178
PINK_SAT_MIN = 55
```

A pixel is classified as healthy pink coral if:

* Its hue is within the pink range.
* Its saturation is high enough.

The code creates a mask containing the detected pink pixels.

```python
pink_mask = (
    (h >= PINK_HUE_MIN)
    &
    (h <= PINK_HUE_MAX)
    &
    (s >= PINK_SAT_MIN)
).astype(np.uint8) * 255
```

---

# 8. Bleached Coral Detection

Bleached coral is represented by the color white.

White pixels usually have:

* Low saturation
* High brightness

The code uses the following thresholds:

```python
WHITE_SAT_MAX = 30
WHITE_VALUE_MIN = 185
```

The white coral mask is created using:

```python
white_mask = (
    (s <= WHITE_SAT_MAX)
    &
    (v >= WHITE_VALUE_MIN)
).astype(np.uint8) * 255
```

This allows the program to separate white coral from the darker background.

---

# 9. Creating the Coral Mask

The pink and white masks are combined.

```python
coral_mask = cv2.bitwise_or(
    pink_mask,
    white_mask
)
```

This produces a complete mask representing all detected coral.

The coral mask contains:

* Healthy pink coral
* Bleached white coral

---

# 10. Noise Removal

Image segmentation may produce small incorrect regions because of:

* Lighting
* Background texture
* Shadows
* Small color variations

To reduce this noise, morphological operations are used.

## Morphological Opening

Opening removes small unwanted pixels.

```python
cv2.MORPH_OPEN
```

## Morphological Closing

Closing fills small holes inside the detected coral region.

```python
cv2.MORPH_CLOSE
```

The code also removes connected components smaller than a specific minimum area.

```python
MIN_BLOB_AREA = 40
```

This helps remove small regions that are likely to be noise.

---

# 11. Comparing Before and After Images

After segmentation and classification, the Before and After images are compared pixel by pixel.

The code creates several masks representing different changes.

---

## Growth Detection

Growth occurs when there was no coral in the Before image but coral appears in the After image.

The logic is:

```text
No Coral Before
+
Coral After
=
Growth
```

The code is:

```python
growth_mask = cv2.bitwise_and(
    no_coral_before,
    coral_after
)
```

---

# 12. Damage Detection

Damage occurs when coral existed in the Before image but is missing in the After image.

The logic is:

```text
Coral Before
+
No Coral After
=
Damage
```

The code is:

```python
damage_mask = cv2.bitwise_and(
    coral_before,
    no_coral_after
)
```

---

# 13. Bleaching Detection

Bleaching occurs when healthy pink coral becomes white.

The logic is:

```text
Pink Before
+
White After
=
Bleaching
```

The code is:

```python
bleach_mask = cv2.bitwise_and(
    pink_before,
    white_after
)
```

---

# 14. Recovery Detection

Recovery occurs when previously bleached white coral becomes healthy pink coral.

The logic is:

```text
White Before
+
Pink After
=
Recovery
```

The code is:

```python
recovery_mask = cv2.bitwise_and(
    white_before,
    pink_after
)
```

---

# 15. Region of Interest

Small alignment errors may cause false changes around the background.

To reduce these errors, the code creates a Region of Interest around the coral colony.

The Region of Interest is based on the combined coral areas from both images.

Only changes near the coral region are considered valid.

This helps prevent background noise from being incorrectly classified as:

* Growth
* Damage
* Bleaching
* Recovery

---

# 16. Visualization

The detected changes are displayed using a color-coded overlay.

The colors are:

| Change         | Color   |
| -------------- | ------- |
| Growth         | Green   |
| Damage / Death | Red     |
| New Bleaching  | Magenta |
| Recovery       | Cyan    |

The detected changes are placed on top of the aligned After image.

A transparency value is used so that the original image can still be seen.

```python
alpha = 0.65
```

The final visualization contains three images:

1. Before image
2. After image aligned with the Before image
3. Changes detected

---

# 17. Statistics

The code calculates the percentage of pixels belonging to each category.

The statistics include:

* Coral area before
* Coral area after
* Growth percentage
* Damage percentage
* New bleaching percentage
* Recovery percentage
* Still healthy percentage
* Still bleached percentage

The percentage is calculated using:

```python
100.0 * np.count_nonzero(mask) / total_pixels
```

These statistics provide a numerical summary of the detected changes.

---

# 18. Final Output

The final output contains:

* Before image
* Aligned After image
* Change detection overlay
* Color legend
* Statistical results

The result is also saved automatically as an image file.

For example:

```text
comparison_coral3_vs_coral4.png
```

---

# 19. Limitations

This solution was tested on two randomly selected images from the dataset.

Therefore, the current implementation should be considered an initial working prototype.

Some limitations include:

1. The HSV thresholds may need adjustment for different lighting conditions.
2. Image alignment depends on finding enough visual features.
3. Large camera angle differences may reduce alignment accuracy.
4. Background colors similar to coral may affect segmentation.
5. The solution has not yet been tested on every image in the complete dataset.

---

# 20. Future Improvements

The solution can be improved in several ways.

Possible improvements include:

* Automatically processing all images in the dataset.
* Automatically finding and pairing Before and After images.
* Using adaptive color thresholds.
* Improving image alignment.
* Using image registration techniques specifically designed for biological images.
* Using machine learning or deep learning for coral segmentation.
* Measuring the actual area of growth and damage.
* Creating a complete report for every coral image pair.

---

# Conclusion

In this project, I developed a Computer Vision solution for analyzing coral colony health by comparing two images taken at different times.

The solution performs the following main tasks:

* Image loading
* Image alignment
* Coral segmentation
* Healthy and bleached coral classification
* Growth detection
* Damage detection
* Bleaching detection
* Recovery detection
* Noise removal
* Visualization
* Statistical analysis

For this submission, I used two randomly selected images to test the complete solution instead of processing the entire database.

The main goal was to first ensure that the complete analysis pipeline works correctly on a sample image pair. The same approach can later be extended to process all image pairs in the dataset automatically.

Although there are still limitations related to lighting, image alignment, and color thresholds, the implemented solution provides a practical starting point for automatically detecting changes in coral colony health using Computer Vision techniques.
