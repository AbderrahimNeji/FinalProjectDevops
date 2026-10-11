"""
Convertir RDD2022 (format PascalVOC XML) en format YOLO
Classes RDD2022: D00 (Longitudinal Crack), D10 (Transverse Crack), D20 (Alligator Crack), D40 (Pothole)
"""

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import random

from src.evaluation.validation import LabelValidationError, xyxy_to_yolo_bbox

# Mapping RDD2022 classes
CLASS_MAPPING = {
    'D00': 0,  # Longitudinal Crack
    'D10': 1,  # Transverse Crack  
    'D20': 2,  # Alligator Crack
    'D40': 3   # Pothole
}

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """Convertir bbox PascalVOC (xmin, ymin, xmax, ymax) en YOLO xywh normalisé."""
    xmin, ymin, xmax, ymax = bbox
    return xyxy_to_yolo_bbox(xmin, ymin, xmax, ymax, img_width, img_height)

def parse_xml_annotation(xml_path):
    """Parser annotation XML PascalVOC"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Dimensions image
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    
    # Extraire les bounding boxes
    annotations = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in CLASS_MAPPING:
            continue
            
        class_id = CLASS_MAPPING[class_name]
        
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        try:
            yolo_bbox = convert_bbox_to_yolo(
                (xmin, ymin, xmax, ymax), img_width, img_height
            )
        except LabelValidationError:
            continue
        annotations.append((class_id, *yolo_bbox))
    
    return annotations

def convert_rdd2022_split(source_dir, dest_dir, split_name):
    """Convertir un split (train/test) du dataset RDD2022"""
    
    images_src = Path(source_dir) / split_name / "images"
    annotations_src = Path(source_dir) / split_name / "annotations" / "xmls"
    
    images_dest = Path(dest_dir) / split_name / "images"
    labels_dest = Path(dest_dir) / split_name / "labels"
    
    images_dest.mkdir(parents=True, exist_ok=True)
    labels_dest.mkdir(parents=True, exist_ok=True)
    
    if not images_src.exists():
        print(f"⚠️  {images_src} n'existe pas, skip {split_name}")
        return 0, 0
    
    # Lister les images
    image_files = list(images_src.glob("*.jpg"))
    converted = 0
    skipped = 0
    
    print(f"\n🔄 Conversion {split_name}...")
    for img_path in tqdm(image_files):
        img_name = img_path.stem
        xml_path = annotations_src / f"{img_name}.xml"
        
        # Copier l'image
        shutil.copy(img_path, images_dest / img_path.name)
        
        # Si annotation existe, convertir
        if xml_path.exists():
            annotations = parse_xml_annotation(xml_path)
            
            # Écrire fichier YOLO .txt
            label_path = labels_dest / f"{img_name}.txt"
            with open(label_path, 'w') as f:
                for class_id, x_center, y_center, width, height in annotations:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            converted += 1
        else:
            # Créer fichier label vide (background)
            label_path = labels_dest / f"{img_name}.txt"
            label_path.touch()
            skipped += 1
    
    print(f"✅ {split_name}: {converted} avec annotations, {skipped} backgrounds")
    return converted, skipped

def create_dataset_yaml(output_dir, dataset_name="RDD2022"):
    """Créer dataset.yaml pour YOLO. Si un split 'val' existe, l'utiliser; sinon 'test'."""
    yaml_path = Path(output_dir) / "dataset.yaml"

    root = Path(output_dir).absolute()
    val_subdir = "val/images" if (root / "val" / "images").exists() else "test/images"

    yaml_content = f"""# RDD2022 Dataset - YOLO Format
# Converted from PascalVOC

path: {root}  # dataset root
train: train/images  # train images (relative to 'path')
val: {val_subdir}     # val images (relative to 'path')

# Classes (RDD2022)
nc: 4  # number of classes
names:
  0: longitudinal_crack  # D00
  1: transverse_crack    # D10
  2: alligator_crack     # D20
  3: pothole             # D40
"""

    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n📄 dataset.yaml créé: {yaml_path}")

def _count_non_empty_labels(labels_dir: Path) -> int:
    """Compter les fichiers de labels non vides (images annotées)."""
    if not labels_dir.exists():
        return 0
    count = 0
    for p in labels_dir.glob("*.txt"):
        try:
            if p.stat().st_size > 0:
                count += 1
        except FileNotFoundError:
            continue
    return count

def create_val_from_train(dest_dir: Path, val_ratio: float = 0.2, seed: int = 42) -> tuple[int, int]:
    """Créer un split 'val' en déplaçant une fraction des images annotées depuis 'train'.
    - Déplace les paires image/label dont le label est non vide.
    - Évite toute fuite en déplaçant (pas de copie) vers 'val/'.
    Retourne (moved_count, remaining_train_annotated).
    """
    train_images = dest_dir / "train" / "images"
    train_labels = dest_dir / "train" / "labels"
    val_images = dest_dir / "val" / "images"
    val_labels = dest_dir / "val" / "labels"

    if not train_images.exists() or not train_labels.exists():
        print("⚠️  Train inexistant, impossible de créer le split val.")
        return 0, 0

    val_images.mkdir(parents=True, exist_ok=True)
    val_labels.mkdir(parents=True, exist_ok=True)

    # Collecter labels non vides
    annotated_labels = [p for p in train_labels.glob("*.txt") if p.stat().st_size > 0]
    if not annotated_labels:
        print("⚠️  Aucun label annoté dans train, split val non créé.")
        return 0, 0

    random.seed(seed)
    k = max(1, int(len(annotated_labels) * val_ratio))
    selected = set(random.sample(annotated_labels, k))

    moved = 0
    for lbl in selected:
        stem = lbl.stem
        # Chercher l'image correspondante (jpg/png)
        img = None
        for ext in (".jpg", ".jpeg", ".png"):
            cand = train_images / f"{stem}{ext}"
            if cand.exists():
                img = cand
                break
        if img is None:
            # pas d'image, ignorer cette entrée
            continue

        # Déplacer image et label
        shutil.move(str(img), str(val_images / img.name))
        shutil.move(str(lbl), str(val_labels / lbl.name))
        moved += 1

    remaining = _count_non_empty_labels(train_labels)
    print(f"\n📦 Split val créé: {moved} images annotées déplacées ({val_ratio*100:.0f}% approx.)")
    print(f"📚 Train annoté restant: {remaining}")
    return moved, remaining

def main():
    print("=" * 70)
    print("  CONVERSION RDD2022 (PascalVOC) → YOLO FORMAT")
    print("=" * 70)
    
    # Chemins - À adapter selon votre environnement
    # Pour Colab: source_dir = Path("/content/RDD2022_Czech") ou Path("/content/Czech")
    # Pour local: source_dir = Path("temp_download/Czech")
    source_dir = Path("temp_download/Czech")  # À modifier si nécessaire
    dest_dir = Path("data/rdd2022_yolo")
    
    # Chercher le dossier automatiquement si non trouvé
    if not source_dir.exists():
        print(f"\n⚠️  {source_dir} n'existe pas!")
        print("\nRecherche du dossier RDD2022 Czech...")

        # Chercher dans les emplacements courants
        possible_locations = [
            Path("Czech"),
            Path("/content/Czech"),
            Path("/content/RDD2022_Czech"),
            Path("RDD2022_Czech"),
            Path("temp_download/RDD2022_Czech")
        ]

        for loc in possible_locations:
            if loc.exists():
                print(f"✅ Trouvé: {loc}")
                source_dir = loc
                break
        else:
            print("\nℹ️  Aucun dossier source trouvé, on utilisera la structure déjà convertie si disponible.")
    
    # Créer structure de destination
    dest_dir.mkdir(parents=True, exist_ok=True)

    train_converted = train_bg = test_converted = test_bg = 0
    # Effectuer conversion uniquement si un dossier source a été trouvé
    if source_dir.exists():
        train_converted, train_bg = convert_rdd2022_split(source_dir, dest_dir, "train")
        test_converted, test_bg = convert_rdd2022_split(source_dir, dest_dir, "test")
    else:
        print("\n🔎 Utilisation des dossiers existants dans:", dest_dir.absolute())
        # Estimer à partir des labels existants
        train_labels = dest_dir / "train" / "labels"
        test_labels = dest_dir / "test" / "labels"
        train_converted = _count_non_empty_labels(train_labels)
        test_converted = _count_non_empty_labels(test_labels)
        # Compter backgrounds (approx: fichiers .txt vides)
        train_bg = len(list(train_labels.glob("*.txt"))) - train_converted if train_labels.exists() else 0
        test_bg = len(list(test_labels.glob("*.txt"))) - test_converted if test_labels.exists() else 0

    # Si le split test est non annoté, créer un split val à partir de train
    val_created = False
    if test_converted == 0:
        print("\n⚠️  Le split 'test' ne contient pas d'annotations. Création d'un split 'val' depuis 'train'...")
        moved, remaining = create_val_from_train(dest_dir, val_ratio=0.2, seed=42)
        val_created = moved > 0

    # Créer dataset.yaml (préférer val si disponible)
    create_dataset_yaml(dest_dir)
    
    print("\n" + "=" * 70)
    print("  ✅ CONVERSION TERMINÉE")
    print("=" * 70)
    print(f"📁 Dataset YOLO: {dest_dir.absolute()}")
    print(f"📊 Train: {train_converted + train_bg} images ({train_converted} annotées)")
    print(f"📊 Test:  {test_converted + test_bg} images ({test_converted} annotées)")
    if (dest_dir / "val" / "images").exists():
        val_labels = dest_dir / "val" / "labels"
        val_annot = _count_non_empty_labels(val_labels)
        val_total = len(list(val_labels.glob("*.txt"))) if val_labels.exists() else 0
        print(f"📊 Val:   {val_total} images ({val_annot} annotées)")
    print(f"\n🚀 Pour entraîner: python simple_train.py")

if __name__ == "__main__":
    main()
