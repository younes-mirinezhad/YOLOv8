# YOLO Training Pipeline

Supports YOLOv8, YOLOv10, YOLOv11, and YOLOv26 detection and segmentation models.

---

## Training a New Model

### 1. Prepare Your Project Files

Copy the `Train/BaseFiles/` folder to a new location for your project (e.g., `Train/MyProject/`). It contains:

- `config.yaml` — main training configuration
- `dataset.yaml` — dataset paths and class names
- `hyperparameters.yaml` — augmentation settings
- `dataset/` — folder structure for images and labels

### 2. Organize Your Dataset

Place your images and labels under the dataset folder following this structure:

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Labels must be in YOLO format (one `.txt` file per image).

### 3. Edit `dataset.yaml`

Set the path to your dataset and define your class names:

```yaml
path: /absolute/path/to/your/dataset
train: images/train
val: images/val
test: images/test

names:
  0: ClassName1
  1: ClassName2
```



### 4. Edit `config.yaml`

Configure the training run. Key fields:


| Field                  | Description                    | Example                        |
| ---------------------- | ------------------------------ | ------------------------------ |
| `model_name`           | Base model checkpoint          | `yolo11n.pt`, `yolo26m-seg.pt` |
| `yaml_path`            | Path to `dataset.yaml`         | `dataset.yaml`                 |
| `hyperparameters_path` | Path to `hyperparameters.yaml` | `hyperparameters.yaml`         |
| `project`              | Output directory for runs      | `runs/detection`               |
| `name`                 | Run name                       | `Train_1`                      |
| `epochs`               | Number of training epochs      | `100`                          |
| `imgsz`                | Input image size               | `640`                          |
| `batch`                | Batch size                     | `32`                           |
| `device`               | GPU id(s) or `cpu`             | `'0'` or `'0,1'`               |
| `lr0`                  | Initial learning rate          | `0.01`                         |
| `patience`             | Early stopping patience        | `100`                          |




### 5. Edit `hyperparameters.yaml` (optional)

Tune data augmentation settings:

```yaml
augment: true
fliplr: 0.5       # left-right flip probability
mosaic: 0.5       # mosaic augmentation probability
mixup: 0.1        # mixup probability
degrees: 5.0      # rotation range
scale: 0.5        # scale range
hsv_h: 0.015      # hue jitter
hsv_s: 0.7        # saturation jitter
hsv_v: 0.4        # brightness jitter
```



### 6. Run Training

Navigate to `Train/` and run:

```bash
python main.py --train --config /path/to/your/config.yaml
```

---



## Exporting a Trained Model

Export immediately after training (uses the just-trained model):

```bash
python main.py --train --config /path/to/config.yaml --export onnx
```

Export a previously trained model by path:

```bash
python main.py --export onnx --model /path/to/best.pt
```

Supported export formats: `onnx`

---



## Scripts


| Script                                  | Description                                             |
| --------------------------------------- | ------------------------------------------------------- |
| `Scripts/convert_LabelMe_To_YOLOdet.py` | Convert LabelMe annotations to YOLO detection format    |
| `Scripts/convert_LabelMe_To_YOLOseg.py` | Convert LabelMe annotations to YOLO segmentation format |
| `Scripts/convert_VGG_To_LabelMe.py`     | Convert VGG annotations to LabelMe format               |
| `Scripts/convertor_VGG_To_YOLOv8.py`    | Convert VGG annotations directly to YOLOv8 format       |
| `Scripts/convertSeg2Det.py`             | Convert segmentation labels to detection format         |
| `Scripts/Train-Test-Split.py`           | Split dataset into train/val/test sets                  |


