# Data Directory

This directory contains all data for the road degradation detection project.

## Structure

```
data/
├── raw/                    # Raw input data
│   ├── videos/            # Video files of roads
│   └── images/            # Image files of roads
├── processed/             # Processed data
│   └── frames/            # Extracted frames from videos
├── annotations/           # YOLO format annotations
│   ├── train/             # Training annotations
│   └── val/               # Validation annotations
└── gps/                   # GPS coordinates data
```

## How to Add Data

### 1. Add Videos

Place your road videos in `raw/videos/`:

```bash
cp your_video.mp4 data/raw/videos/
```

### 2. Add Images

Place road images in `raw/images/`:

```bash
cp your_image.jpg data/raw/images/
```

### 3. Extract Frames

Extract frames from videos:

```bash
python src/data_preparation/extract_frames.py
```

### 4. Annotate Data

Use tools like LabelImg or Roboflow to create YOLO annotations:

- Format: One .txt file per image
- Classes: 0=pothole, 1=crack, 2=crazing, 3=faded_marking
- Save to `annotations/train/` or `annotations/val/`

### 5. Add GPS Data

GPS coordinates in JSON format in `gps/` directory.

## Dataset Recommendations

Public datasets for road degradation:

- **RDD2020** - Road Damage Dataset
- **Pothole Dataset** - Various pothole images
- **Crack Detection Dataset** - Pavement crack dataset

## Data Format

### YOLO Annotation Format

Each image has a corresponding .txt file with:

```
<class> <x_center> <y_center> <width> <height>
```

Values normalized to 0-1.

Example:

```
0 0.5 0.5 0.3 0.2
1 0.7 0.8 0.15 0.1
```

### GPS Data Format

```json
{
  "frame_id": "frame_001.jpg",
  "latitude": 48.8566,
  "longitude": 2.3522,
  "timestamp": "2026-01-09T14:30:00Z"
}
```

## Classes

- **0**: Pothole
- **1**: Longitudinal Crack
- **2**: Crazing
- **3**: Faded Road Marking
