***

# Golf Ball Launch Detection and Analysis

An AI-powered golf ball tracking and launch performance analyser combining custom-trained Ultralytics YOLOv8 detection with physics-based trajectory calculations. Detects the golf ball in swing videos, tracks its movement, and estimates launch velocity, angle, flight time, apex height, and carry distance.

***

## Features

- Custom-trained YOLOv8 model for highly accurate golf ball detection.
- Interactive region of interest (ROI) selection for focused tracking.
- Calculations of ball kinematics and projectile flight parameters.
- Real-time 600x600 preview window during video analysis.
- Saves annotated output videos showing detected balls and trajectories.
- Supports OpenVINO accelerated inference when available.
- Full command-line interface for flexible input and settings.
- Modular, documented Python code with no dependencies beyond standard packages.

***

## Prerequisites

- Python 3.8 or later (64-bit recommended).
- pip (the Python package installer).
- A modern CPU; GPU optional but can accelerate processing.
- Cloned repository including model weights and sample test videos.

***

## Setup Instructions

### 1. Clone this repository

```bash
git clone https://github.com/MitulAggarwal1/Golf-Launch-Monitor.git
cd Golf-Launch-Monitor
```

### 2. Create and activate a Python virtual environment

On Windows:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Prepare model weights

- The pre-trained model weights `best.pt` and optional OpenVINO export folder `best_openvino_model` are included in the `weights/` directory.
- These weights are custom-trained for golf ball detection and are ready for immediate use.
- No retraining is required to run the project inference.
- The code is configured to use relative paths to these weights.

***

## Dataset Information

- The golf ball detection model was trained on a custom dataset created by manually labelling golf ball images.
- The complete dataset is hosted on Roboflow and can be accessed here: [https://app.roboflow.com/mitul-aggarwal-scv7i/golfball-yyex6/1](https://app.roboflow.com/mitul-aggarwal-scv7i/golfball-yyex6/1)
- Use the Roboflow link for dataset inspection or to generate new versions for training if desired.

***

## Running the Project

Run the main analysis script with a video input:

```bash
python main.py
```

- Follow the prompts to select the region of interest (ROI) in the video.
- The script will display a 600x600 preview with ball detection and tracking.
- An annotated output video named `ball_annotated.mp4` will be saved in the project directory after processing.

***

## Troubleshooting

- Ensure the virtual environment is activated before running.
- Check that all dependencies are installed via the requirements file.
- Verify that the `weights` directory and `best.pt` file exist in the project.
- Confirm that the video file path supplied exists and is readable.
- Use OpenVINO inference only if the `best_openvino_model` folder is present; otherwise, the script falls back to the PyTorch weights by default.

***

## Repository Structure

```
Golf-Launch-Monitor/
│
├── main.py                   # Inference and launch analysis script
├── requirements.txt          # Python dependencies
├── data.yaml                 # Dataset config for YOLO training
├── weights/                  # Trained model weights folder
│   ├── best.pt
│   └── best_openvino_model/  # Optional OpenVINO export
├── test_videos/              # Sample test videos provided for evaluation
├── README.dataset.txt        # Roboflow dataset description
├── README.roboflow.txt       # Roboflow project info
├── README.md                 # This file
└── .gitignore                # To exclude virtual envs, video outputs, etc.
```

***

## Acknowledgments

- Ultralytics YOLO framework for object detection.
- OpenCV for efficient video processing.
- Roboflow for dataset management and annotation platform.

***

## License

Please contact the author for collaboration or reuse.

***
