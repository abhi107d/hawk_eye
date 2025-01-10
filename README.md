# Hawk eye (exam hall cheating detection system) 

## Overview
This project aims to develop a system for exam hall cheating detection It utilizes LSTM (Long Short-Term Memory) networks for gesture recognition and MediaPipe for hand tracking and landmark detection.

## Algorithm
 - First we track persons using deepsort and yolo
 - pass each detections through pose detection model 
 - Extract pose landmarks
 - collect landmarks for n number of frames for each persons 
 - pass landmark sequence to classfier model to get peredection (cheatin/non cheating)
 - show aleart (red box)

## Requirements
- Python 3.11
- PyTorch
- CUDA Toolkit (11.8)
- cuDNN (8.9.7)
- ultralytics
- OpenCV
- Additional Python Libraries (scikit-learn, huggingface_hub, etc.)

## Instalation

### Step 1: Clone the Repository
Clone the project repository to your local machine:
```bash
git clone https://github.com/abhi107d/hawk_eye.git
cd hawk_eye
```

### Step 2: Create a Conda Environment 
Use the following command to create a Conda environment in a specified directory:
```bash
conda create -p env python=3.11
```

### Step 3: Activate the Environment
Activate the newly created Conda environment:
```bash
conda activate ./env
```

### Step 4: Install CUDA Toolkit and cuDNN
Install the CUDA Toolkit and cuDNN required for GPU acceleration
refer: https://developer.nvidia.com/rdp/cudnn-archive for version compatblity of cudnn:
```bash
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.9.7
```

### Step 5: Install PyTorch with GPU Support
Install PyTorch, torchvision, and torchaudio along with GPU support
refer: https://pytorch.org/get-started/locally/ 
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Step 6: Install Additional Dependencies
Install the remaining dependencies required for the project:
```bash
pip -r requirements.txt
```

## Usage
After completing the installation, navigate to the project directory and activate the Conda environment to begin using the tool.

```bash
cd hawk_eye
conda activate ./env
python app.py
```

## Training

### Step 1:
```bash
python python dataset_builder.py --videosrc "../videos_test/cheating.mp4"  --label 1 --show True #example usage
```
-- videosrc = sorce video
-- label = 1 for cheating class and 2 for non cheating class
-- show = if True shows video and detections 
-- seqlen = sequence length of video default 20

### Step 2:
Run Trainer/Trainer.ipynb

## Notes
- Ensure that you have a compatible GPU and drivers installed for CUDA Toolkit and cuDNN.
- model is not accurate we need to collect more data

## :TODO
- Develop UI
- Aleart system 

## Acknowledgments
- PyTorch: [https://pytorch.org/](https://pytorch.org/)
- YOLO



