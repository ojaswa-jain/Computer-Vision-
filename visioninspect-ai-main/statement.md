# Project Statement: VisionInspect-AI

## 1. Problem Statement
In high-throughput manufacturing environments (such as semiconductor fabrication, metal stamping, precision optics, and textile production), surface defect inspection is critical to product safety, compliance, and cost minimization. Traditional manual quality inspection by human operators suffers from significant limitations:
- **Fatigue and Inconsistency**: Human visual acuity degrades over prolonged shifts, resulting in defect miss rates up to 15-30%.
- **Throughput Bottlenecks**: Manual inspection cannot keep pace with high-speed automated assembly lines operating at dozens of units per second.
- **Subjectivity and Lack of Traceability**: Visual grading criteria vary across operators, making root-cause analysis and defect metrology difficult to audit.
- **Latency**: Inspection feedback arrives too late to adjust tooling or process parameters before large batches are ruined.

Existing commercial machine vision solutions are often rigid, closed-source black boxes that require complex proprietary hardware, high license fees, and lack transparent algorithmic explainability. **VisionInspect-AI** resolves this challenge by providing an open, modular, reproducible, and computationally efficient Computer Vision system that automates surface defect detection, morphological segmentation, geometric metrology, and tolerance-based quality grading using both classical and frequency-domain computer vision techniques.

---

## 2. Scope of the Project
The scope of VisionInspect-AI covers:
- **Multi-Surface Defect Inspection**: Automated identification of structural flaws including hairline cracks, abrasive scratches, localized pitting/voids, surface discoloration, and foreign particulate inclusions across metal, ceramic, silicon, and composite substrates.
- **Illumination & Artifact Correction**: Robust pre-processing algorithms (CLAHE, bilateral edge-preserving smoothing, multi-channel color space decomposition) to handle non-uniform industrial factory lighting and specular reflections.
- **Texture Anomaly Decomposition**: 2D Gabor wavelet filter banks to segment subtle surface irregularities and structural weave discrepancies from periodic background textures.
- **Dimensional Metrology & Calibration**: Precise measurement of defect physical parameters (length, width, oriented bounding box, perimeter, area, convex hull depth, and aspect ratio) calibrated from pixel coordinates to physical metric units (millimeters).
- **Automated Pass/Fail Grading**: Configurable industrial tolerance thresholds compliant with standard manufacturing quality specifications (e.g., ISO 9001 / IPC-A-610 standards).
- **Headless CLI & Batch Execution**: Comprehensive command-line interface supporting single-image analysis, high-speed batch directory processing, quantitative ground truth benchmarking (IoU, Dice, Precision, Recall), and export to structured JSON/CSV logs.
- **Automated Diagnostic Reporting**: Automated generation of publication-ready audit reports, side-by-side diagnostic cards, and defect heatmaps.

### Exclusions / Out of Scope
- Direct hardware driver interfacing with industrial GigE Vision or GenICam cameras (images and video frames are ingested via standard file interfaces or streams).
- Closed-loop robotic actuator control for physical part ejection (actuation triggers are communicated via output logs and structured status codes).

---

## 3. Target Users
- **Quality Assurance & Control (QA/QC) Engineers**: Professionals responsible for defining tolerance limits, auditing manufacturing yield, and reviewing defect classifications.
- **Industrial Automation Integrators**: Engineers deploying headless computer vision inspection scripts onto edge computing units (e.g., Raspberry Pi, NVIDIA Jetson, industrial PCs) on factory conveyor lines.
- **Manufacturing Plant Operators**: Personnel reviewing daily defect analytics, yield trends, and batch compliance reports.
- **Computer Vision Researchers & Students**: Academics and students studying classical image processing, morphological transforms, texture filtering, and quantitative computer vision evaluation methodologies.

---

## 4. High-Level Features
1. **Adaptive Image Conditioning & Illumination Normalization**:
   - Contrast Limited Adaptive Histogram Equalization (CLAHE) in CIE $L^*a^*b^*$ and HSV color spaces.
   - Bilateral filtering preserving sharp defect boundaries while attenuating high-frequency sensor noise.
   - Gamma correction for underexposed and overexposed factory captures.

2. **Multi-Scale Morphological & Frequency Defect Isolation**:
   - Dual Top-Hat (bright anomaly) and Black-Hat (dark anomaly) morphological transforms using tailored structuring elements.
   - 2D multi-orientation Gabor filter bank decomposition for surface texture defect isolation.
   - Dynamic Otsu and adaptive Sauvola binarization for robust thresholding without manual per-image tuning.

3. **Intelligent Geometric Classification Engine**:
   - Rule-based decision forest classifying contours based on compactness, eccentricity, aspect ratio, solidity, and chromatic delta into distinct defect taxonomies (*Crack*, *Scratch*, *Pit/Void*, *Discoloration*, *Foreign Particle*).

4. **Calibrated Dimensional Metrology**:
   - Sub-pixel contour boundary tracking with calibrated pixel-to-millimeter scaling.
   - Computation of oriented bounding boxes (minimum area rectangles), equivalent diameter, major/minor axis lengths, and perimeter.
   - Severity classification into *Pass*, *Minor*, *Major*, and *Critical* based on tolerance rules.

5. **Diagnostic Visual Overlays & HUD**:
   - Multi-layer visual annotator rendering color-coded contours, oriented bounding boxes, center centroids, and diagnostic metadata overlays.
   - Quad-view comparison panel: Original Image, Preprocessed Normalization, Anomaly Probability Heatmap, and Final Annotated Inspection View.

6. **Automated Batch Pipeline & Auditing**:
   - High-throughput headless CLI capable of batch processing thousands of parts with detailed CSV/JSON telemetry export.
   - Ground truth evaluation engine calculating Pixel IoU, Dice Coefficient, Detection Precision, Recall, and F1-Score.
   - Automated PDF/HTML Quality Inspection Report generator.
