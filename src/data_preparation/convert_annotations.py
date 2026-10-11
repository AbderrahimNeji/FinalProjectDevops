"""
Convert annotations from various formats to YOLO format.
Supports COCO, Pascal VOC, and CSV formats.
"""

import json
import os
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm

from src.evaluation.validation import LabelValidationError, xyxy_to_yolo_bbox


# Class mapping
CLASSES = {
    0: "pothole",
    1: "longitudinal_crack",
    2: "crazing",
    3: "faded_marking"
}

CLASS_TO_ID = {v: k for k, v in CLASSES.items()}


class AnnotationConverter:
    """Convert annotations to YOLO format."""
    
    def __init__(self, input_dir, output_dir, format_type='coco'):
        """
        Initialize converter.
        
        Args:
            input_dir: Directory with source annotations
            output_dir: Directory for YOLO format annotations
            format_type: Source format ('coco', 'voc', 'csv')
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.format_type = format_type
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def convert_from_coco(self, annotation_file):
        """Convert COCO JSON to YOLO format."""
        print("🔄 Converting from COCO format...")
        
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create image id to filename mapping
        images = {img['id']: img for img in coco_data['images']}
        
        # Group annotations by image
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # Convert each image's annotations
        for img_id, annotations in tqdm(annotations_by_image.items()):
            img_info = images[img_id]
            img_width = img_info['width']
            img_height = img_info['height']
            filename = Path(img_info['file_name']).stem
            
            yolo_annotations = []
            for ann in annotations:
                # COCO bbox: [x, y, width, height]
                x, y, w, h = ann['bbox']
                category_id = ann['category_id']
                try:
                    x_center, y_center, w_norm, h_norm = xyxy_to_yolo_bbox(
                        x, y, x + w, y + h, img_width, img_height
                    )
                except LabelValidationError:
                    continue
                yolo_annotations.append(
                    f"{category_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                )
            
            # Write YOLO annotation file
            output_file = self.output_dir / f"{filename}.txt"
            with open(output_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))
        
        print(f"✅ Converted {len(annotations_by_image)} annotations")
    
    def convert_from_voc(self):
        """Convert Pascal VOC XML to YOLO format."""
        print("🔄 Converting from Pascal VOC format...")
        
        xml_files = list(self.input_dir.glob('*.xml'))
        
        for xml_file in tqdm(xml_files):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get image dimensions
            size = root.find('size')
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)
            
            yolo_annotations = []
            
            # Process each object
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                
                # Skip if class not in our mapping
                if class_name not in CLASS_TO_ID:
                    continue
                
                class_id = CLASS_TO_ID[class_name]
                
                # Get bounding box
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)

                try:
                    x_center, y_center, width, height = xyxy_to_yolo_bbox(
                        xmin, ymin, xmax, ymax, img_width, img_height
                    )
                except LabelValidationError:
                    continue
                yolo_annotations.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )
            
            # Write YOLO annotation file
            output_file = self.output_dir / f"{xml_file.stem}.txt"
            with open(output_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))
        
        print(f"✅ Converted {len(xml_files)} annotations")
    
    def convert_from_csv(self, csv_file):
        """
        Convert CSV to YOLO format.
        Expected CSV columns: filename, class, xmin, ymin, xmax, ymax, width, height
        """
        print("🔄 Converting from CSV format...")
        
        df = pd.read_csv(csv_file)
        
        # Group by filename
        grouped = df.groupby('filename')
        
        for filename, group in tqdm(grouped):
            yolo_annotations = []
            
            # Get image dimensions from first row
            img_width = group.iloc[0]['width']
            img_height = group.iloc[0]['height']
            
            for _, row in group.iterrows():
                class_name = row['class']
                
                # Skip if class not in our mapping
                if class_name not in CLASS_TO_ID:
                    continue
                
                class_id = CLASS_TO_ID[class_name]
                
                # Get bounding box
                xmin = row['xmin']
                ymin = row['ymin']
                xmax = row['xmax']
                ymax = row['ymax']

                try:
                    x_center, y_center, width, height = xyxy_to_yolo_bbox(
                        xmin, ymin, xmax, ymax, img_width, img_height
                    )
                except LabelValidationError:
                    continue
                yolo_annotations.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )
            
            # Write YOLO annotation file
            output_file = self.output_dir / f"{Path(filename).stem}.txt"
            with open(output_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))
        
        print(f"✅ Converted {len(grouped)} annotations")
    
    def convert(self, annotation_file=None):
        """Run conversion based on format type."""
        if self.format_type == 'coco':
            if not annotation_file:
                annotation_file = self.input_dir / 'annotations.json'
            self.convert_from_coco(annotation_file)
        elif self.format_type == 'voc':
            self.convert_from_voc()
        elif self.format_type == 'csv':
            if not annotation_file:
                annotation_file = self.input_dir / 'annotations.csv'
            self.convert_from_csv(annotation_file)
        else:
            raise ValueError(f"Unsupported format: {self.format_type}")


def create_class_file(output_path):
    """Create classes.txt file for YOLO."""
    with open(output_path, 'w') as f:
        for class_id in sorted(CLASSES.keys()):
            f.write(f"{CLASSES[class_id]}\n")
    print(f"✅ Created classes file: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert annotations to YOLO format')
    parser.add_argument('--input', type=str, required=True, help='Input directory with annotations')
    parser.add_argument('--output', type=str, required=True, help='Output directory for YOLO annotations')
    parser.add_argument('--format', type=str, choices=['coco', 'voc', 'csv'], default='coco',
                        help='Source annotation format')
    parser.add_argument('--annotation-file', type=str, default=None,
                        help='Annotation file (for COCO/CSV formats)')
    
    args = parser.parse_args()
    
    converter = AnnotationConverter(
        input_dir=args.input,
        output_dir=args.output,
        format_type=args.format
    )
    
    converter.convert(args.annotation_file)
    
    # Create classes file
    create_class_file(Path(args.output) / 'classes.txt')


if __name__ == "__main__":
    main()
