# 🎉Eleflai🎉
## It only takes a few simple steps to master the use of Eleflai.🤳
Eleflai is an AI-assisted otoscopy system comprising a mobile otoscope, a companion app, cloud-based analysis, and an image-analysis model. This repository provides model training scripts and an offline inference demonstration notebook. The mobile app and cloud backend source code are not included in this repository.

https://github.com/user-attachments/assets/a04c8065-e904-4943-b376-f081aa455f6b

## Eleflai is an AI-powered wireless otoscope designed to enhance ear health management in primary care settings.
![Eleflai-demo](https://github.com/user-attachments/assets/558222f6-e412-4465-adf4-5fd32cc773c3)
# Model Architecture🤖
## YOLOV8-architecture
<img width="1000" height="1047" alt="yolov8-architecture" src="https://github.com/user-attachments/assets/85f97941-3f84-421d-b47e-a3aa7f922356" />

## YOLOV9-architecture
<img width="2555" height="1645" alt="yolov9-architecture" src="https://github.com/user-attachments/assets/b83e0390-3ae6-43a3-b25e-5106548df0c1" />

## YOLOV10-architecture
<img width="1080" height="1186" alt="yolov10-architecture" src="https://github.com/user-attachments/assets/6975ac0b-116e-48be-b59c-56ac1f7c7154" />

## Repository contents

- `Eleflai_inference_workflow_demo.ipynb`: an offline inference demonstration using previously acquired otoscopic images.
- `yolov8_finetune.py`, `yolov9_finetune.py`, and `yolov10_finetune.py`: model fine-tuning scripts.
- `SparK/`: code for the self-supervised pretraining component.

## Hardware requirements

The offline inference notebook processes existing image files and does not require the Eleflai mobile otoscope. The deployed clinical workflow uses the Eleflai mobile otoscope, companion app, network connection, and cloud-based analysis service. The app and cloud backend source code are not included in this repository.

## Offline inference notebook

The notebook specifies the following default paths:

- Dataset configuration: `demo_assets/github_demo_dataset.yaml`
- Model checkpoint: `weights/eleflai_yolo_demo.pt`
- Output directory: `demo_outputs/`

Users can set `DEVICE` to `"cpu"` for CPU inference or `"cuda:0"` for a compatible CUDA GPU. They can specify their own image paths using `IMAGE_PATHS`, or provide a dataset YAML file and change `DATA_YAML_PATH`.

The notebook is designed to save detection results as a CSV file, a JSON file, and images with predicted bounding boxes in `demo_outputs/`.

## Installation guide

Clone the repository:

```bash
git clone https://github.com/YubiaoYue/Eleflai.git
cd Eleflai
```

### YOLO fine-tuning and offline inference

Create a separate Python environment:

```bash
conda create -n eleflai-yolo python=3.10 -y
conda activate eleflai-yolo
```

Install the PyTorch build appropriate for your operating system and CPU/CUDA configuration, following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/). Then install the packages used by the YOLO scripts and demonstration notebook:

```bash
python -m pip install ultralytics==8.3.29 pillow pandas matplotlib notebook
python -m pip check
```

To open the offline demonstration:

```bash
python -m notebook Eleflai_inference_workflow_demo.ipynb
```

The notebook additionally requires input images and a compatible inference checkpoint at the paths specified in the notebook.

### SparK pretraining

SparK pretraining uses a separate environment. See `SparK/INSTALL.md` for the original SparK dependency recommendations. Install the PyTorch build verified for the pretraining hardware, followed by the dependencies listed in this repository:

```bash
conda create -n eleflai-spark python=[VERIFIED_PYTHON_VERSION] -y
conda activate eleflai-spark
# Install the verified PyTorch and torchvision versions here.
python -m pip install -r SparK/pretrain/requirements.txt
python -m pip check
```
