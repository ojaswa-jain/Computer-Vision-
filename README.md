VisionInspect-AI: Industrial Surface Defect Detection & Metrology Pipeline
Python 3.11 OpenCV Tests License: MIT

An end-to-end Computer Vision system designed for automated industrial surface quality control, defect segmentation, dimensional metrology, and tolerance-based Pass/Fail grading.

1. Project Overview
In modern manufacturing, manual visual inspection of surfaces (such as stamped metal, printed circuit boards, ceramic tiles, and textiles) is slow, subjective, and prone to operator fatigue. VisionInspect-AI provides a fully automated, command-line executable Computer Vision pipeline that:

Ingests surface images and normalizes non-uniform illumination.
Segments microscopic and macroscopic flaws (cracks, scratches, voids/pits, particles).
Computes calibrated physical measurements (length, width, area in millimeters).
Classifies defects into standardized industrial taxonomies.
Enforces configurable tolerance thresholds to output an automated PASS / FAIL verdict.
Generates comprehensive visual diagnostic cards, defect heatmaps, JSON telemetry, and CSV audit logs.
2. Key Features
Pure CLI Executability: Operates seamlessly in headless server and CI/CD environments without requiring a GUI.
Robust Illumination Normalization: Uses CLAHE and bilateral filtering to eliminate factory lighting gradients and sensor noise while preserving sharp crack boundaries.
Dual Morphological Filtering: Isolates dark defects (cracks, pits) via Black-Hat transforms and bright defects (particulate inclusions, gouges) via Top-Hat transforms.
Geometric Metrology: Calibrates pixel coordinates to physical metric dimensions (
m
m
 and 
m
m
2
) using minimum area rotated bounding boxes.
Tolerance-Based Quality Grading: Evaluates defects against strict industrial standards into PASS, MINOR, MAJOR, and CRITICAL severities.
Diagnostic Quad-Panel: Exports side-by-side visual panels (Original Input, Enhanced Image, Anomaly Heatmap, Annotated HUD).
Comprehensive Benchmarking: Computes academic metrics (Pixel IoU, Dice/F1, Precision, Recall) against ground truth masks.
3. Technologies Used
Language: Python 3.11+
Computer Vision: OpenCV (opencv-python)
Scientific Computing: NumPy, SciPy
Data Analysis & Export: Pandas
Visualization: Matplotlib, PIL
Testing Framework: Pytest
4. Installation & Environment Setup
Step 1: Clone the Repository
git clone https://github.com/<your-github-username>/vit_project.git
cd vit_project
Step 2: Set Up Virtual Environment (Recommended)
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
Step 3: Install Dependencies
python3 -m pip install -r requirements.txt
5. Execution Guide
5.1 Generate Synthetic Benchmark Samples
To create realistic industrial surface samples (steel crack, PCB scratch, ceramic pit, clean reference) and ground-truth annotation masks:

python3 generate_samples.py
5.2 Single Image Inspection
Inspect a single image and generate annotated visual outputs and telemetry JSON:

python3 main.py inspect --image samples/steel_plate_crack.png --output-dir output/inspect_steel/
Output Artifacts Generated in output/inspect_steel/:

steel_plate_crack_annotated.png (Color-coded bounding boxes and HUD)
steel_plate_crack_diagnostic_panel.png (2x2 comparison card)
steel_plate_crack_defect_mask.png (Binary segmented mask)
steel_plate_crack_anomaly_heatmap.png (Normalized anomaly map)
steel_plate_crack_telemetry.json (Structured JSON defect report)
5.3 Batch Directory Processing
Run high-throughput inspection on a directory of images:

python3 main.py batch --input-dir samples/ --output-dir output/batch_results/
Generates individual annotated images, a consolidated batch_inspection_summary.csv, and an executive audit batch_inspection_report.md.

5.4 Quantitative Evaluation Benchmark
Evaluate segmentation predictions against ground truth masks to calculate Pixel IoU, Dice/F1, Precision, and Recall:

python3 main.py evaluate --images-dir samples/ --masks-dir samples/masks/ --output-dir output/evaluation/
5.5 Latency and Throughput Benchmark
Benchmark execution latency and Frames Per Second (FPS):

python3 main.py benchmark --runs 30 --size 512
6. Running Tests
The project contains 16 automated unit and integration tests covering preprocessing, segmentation, metrology, and CLI commands.

To run the complete test suite:

python3 -m pytest tests/ -v
Expected output:

============================== 16 passed in 0.88s ==============================
7. Project Architecture & Directory Structure
vit_project/
├── .gitignore                      # Git exclusion rules
├── README.md                       # Comprehensive setup and execution documentation
├── statement.md                    # Problem statement and project scope
├── requirements.txt                # Python dependencies
├── main.py                         # Master CLI entrypoint (inspect, batch, evaluate, benchmark)
├── generate_samples.py             # Synthetic industrial sample generator
├── generate_report_pdf.py          # Script compiling the 15-section project report PDF
├── visioninspect/                  # Core package
│   ├── __init__.py
│   ├── config.py                   # Dataclasses and inspection thresholds
│   ├── pipeline.py                 # Master inspection pipeline
│   ├── evaluation.py               # Benchmark evaluation engine (IoU, Dice, Precision, Recall)
│   ├── preprocessing/              # Illumination correction & color transforms
│   │   ├── __init__.py
│   │   ├── enhancement.py          # CLAHE, bilateral filter, gamma correction
│   │   └── color_spaces.py         # CIE LAB, HSV, chromatic delta
│   ├── features/                   # Feature extraction
│   │   ├── __init__.py
│   │   ├── edges_contours.py       # Canny edges, morphological gradients, contour descriptors
│   │   └── texture_gabor.py        # 2D Gabor wavelet filter bank
│   ├── detector/                   # Defect segmentation and classification
│   │   ├── __init__.py
│   │   ├── morphological_detector.py # Dual Top-Hat / Black-Hat segmentation
│   │   └── classifier.py           # Geometric and photometric defect classifier
│   ├── metrology/                  # Metric calibration & tolerance grading
│   │   ├── __init__.py
│   │   └── measurement.py          # Pixel-to-mm scaling, severity grading, summary
│   ├── visualization/              # Visual diagnostic renderers
│   │   ├── __init__.py
│   │   ├── annotator.py            # HUD overlay and color-coded bounding annotations
│   │   └── summary_card.py         # Multi-view 2x2 comparison panels
│   └── reporting/                  # Export utilities
│       ├── __init__.py
│       └── exporter.py             # JSON, CSV, and Markdown audit export
├── docs/                           # Documentation and submission report
│   ├── PROJECT_REPORT.md           # Full 15-section academic report
│   └── PROJECT_REPORT.pdf          # Formatted PDF report for portal upload
├── samples/                        # Test samples and ground truth masks
│   └── masks/
└── tests/                          # Automated unit and integration test suite
    ├── __init__.py
    ├── test_cli.py
    ├── test_detector.py
    ├── test_metrology.py
    └── test_preprocessing.py
