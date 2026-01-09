"""
Script pour générer des images synthétiques de dégradations routières.
Crée des images réalistes avec potholes, fissures, faïençage et marquages effacés.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import json
import random

def create_road_background(width=1920, height=1080):
    """Crée un fond de route réaliste."""
    
    # Créer une base asphaltée
    img = Image.new('RGB', (width, height), color=(60, 60, 65))
    pixels = img.load()
    
    # Ajouter du bruit pour texture réaliste
    for y in range(height):
        for x in range(width):
            noise = random.randint(-10, 10)
            r, g, b = pixels[x, y]
            pixels[x, y] = (
                max(0, min(255, r + noise)),
                max(0, min(255, g + noise)),
                max(0, min(255, b + noise))
            )
    
    # Ajouter des variations de texture
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    
    return img

def add_pothole(img, x, y, size=100):
    """Ajoute un nid-de-poule à l'image."""
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Ombre du nid-de-poule
    shadow_color = (20, 20, 25, 150)
    draw.ellipse(
        [x - size//2, y - size//2, x + size//2, y + size//2],
        fill=shadow_color
    )
    
    # Bord du nid-de-poule
    edge_color = (40, 40, 45, 200)
    draw.ellipse(
        [x - size//2, y - size//2, x + size//2, y + size//2],
        outline=edge_color,
        width=3
    )
    
    return img

def add_crack(img, start_x, start_y, length=300, angle=0):
    """Ajoute une fissure à l'image."""
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Calculer les points de fin
    end_x = start_x + int(length * np.cos(np.radians(angle)))
    end_y = start_y + int(length * np.sin(np.radians(angle)))
    
    # Fissure principale
    crack_color = (30, 30, 35, 180)
    draw.line(
        [(start_x, start_y), (end_x, end_y)],
        fill=crack_color,
        width=3
    )
    
    # Ajouter des ramifications
    for i in range(3):
        branch_x = start_x + int((length // 3) * (i + 1) * np.cos(np.radians(angle)))
        branch_y = start_y + int((length // 3) * (i + 1) * np.sin(np.radians(angle)))
        
        branch_angle = angle + random.uniform(-20, 20)
        branch_length = length // 4
        
        branch_end_x = branch_x + int(branch_length * np.cos(np.radians(branch_angle)))
        branch_end_y = branch_y + int(branch_length * np.sin(np.radians(branch_angle)))
        
        draw.line(
            [(branch_x, branch_y), (branch_end_x, branch_end_y)],
            fill=crack_color,
            width=2
        )
    
    return img

def add_crazing(img, center_x, center_y, area_size=300):
    """Ajoute du faïençage (réseau de fissures) à l'image."""
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Créer un réseau de fissures
    crack_color = (35, 35, 40, 160)
    
    for _ in range(15):
        x1 = center_x + random.randint(-area_size//2, area_size//2)
        y1 = center_y + random.randint(-area_size//2, area_size//2)
        x2 = x1 + random.randint(-50, 50)
        y2 = y1 + random.randint(-50, 50)
        
        draw.line([(x1, y1), (x2, y2)], fill=crack_color, width=2)
    
    return img

def add_faded_marking(img, x, y, width=400, height=30):
    """Ajoute un marquage routier effacé à l'image."""
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Marquage effacé en blanc semi-transparent
    marking_color = (200, 200, 200, 80)
    draw.rectangle([x, y, x + width, y + height], fill=marking_color)
    
    # Ajouter un dégradé pour plus de réalisme
    for i in range(height):
        alpha = int(80 * (1 - i / height))
        color = (200, 200, 200, alpha)
        draw.rectangle([x, y + i, x + width, y + i + 1], fill=color)
    
    return img

def generate_pothole_image(filename, width=1920, height=1080):
    """Génère une image avec nid-de-poule."""
    img = create_road_background(width, height)
    
    # Ajouter 2-3 nids-de-poule
    num_potholes = random.randint(2, 3)
    for _ in range(num_potholes):
        x = random.randint(300, width - 300)
        y = random.randint(300, height - 300)
        size = random.randint(60, 150)
        add_pothole(img, x, y, size)
    
    img.save(filename, quality=95)
    print(f"✓ Créée: {filename}")

def generate_crack_image(filename, width=1920, height=1080):
    """Génère une image avec fissures."""
    img = create_road_background(width, height)
    
    # Ajouter 2-3 fissures
    num_cracks = random.randint(2, 3)
    for _ in range(num_cracks):
        x = random.randint(200, width - 200)
        y = random.randint(200, height - 200)
        angle = random.randint(-45, 45)
        add_crack(img, x, y, length=random.randint(200, 400), angle=angle)
    
    img.save(filename, quality=95)
    print(f"✓ Créée: {filename}")

def generate_crazing_image(filename, width=1920, height=1080):
    """Génère une image avec faïençage."""
    img = create_road_background(width, height)
    
    # Ajouter faïençage
    x = random.randint(400, width - 400)
    y = random.randint(400, height - 400)
    add_crazing(img, x, y, area_size=random.randint(250, 400))
    
    img.save(filename, quality=95)
    print(f"✓ Créée: {filename}")

def generate_marking_image(filename, width=1920, height=1080):
    """Génère une image avec marquages effacés."""
    img = create_road_background(width, height)
    
    # Ajouter 2-3 marquages effacés
    num_markings = random.randint(2, 3)
    for _ in range(num_markings):
        x = random.randint(200, width - 600)
        y = random.randint(200, height - 100)
        add_faded_marking(img, x, y, width=random.randint(300, 500), height=random.randint(20, 40))
    
    img.save(filename, quality=95)
    print(f"✓ Créée: {filename}")

def generate_mixed_image(filename, width=1920, height=1080):
    """Génère une image avec plusieurs types de dégradations."""
    img = create_road_background(width, height)
    
    # Ajouter plusieurs types de dégradations
    add_pothole(img, 400, 400, 80)
    add_crack(img, 900, 600, 250, angle=30)
    add_faded_marking(img, 500, 200, 400, 30)
    add_crazing(img, 1400, 800, 200)
    
    img.save(filename, quality=95)
    print(f"✓ Créée: {filename}")

def create_annotations(image_filename, annotations_data):
    """Crée les annotations YOLO pour une image."""
    base_name = Path(image_filename).stem
    annotation_path = Path("data/annotations/train") / f"{base_name}.txt"
    
    with open(annotation_path, 'w') as f:
        f.write("# Format YOLO: class x_center y_center width height (normalized 0-1)\n")
        for annotation in annotations_data:
            f.write(f"{annotation}\n")

def main():
    """Fonction principale."""
    
    print("=" * 70)
    print("🚗 Génération d'images synthétiques de dégradations routières")
    print("=" * 70)
    
    # Créer le dossier s'il n'existe pas
    images_dir = Path("data/raw/images")
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Supprimer les fichiers .txt existants
    for txt_file in images_dir.glob("*.txt"):
        txt_file.unlink()
        print(f"🗑️  Supprimé: {txt_file}")
    
    print("\n📸 Génération des images...\n")
    
    # Générer les images
    images_data = [
        ("sample_pothole_001.jpg", generate_pothole_image, [
            "0 0.4 0.4 0.2 0.25",
            "0 0.7 0.6 0.15 0.2"
        ]),
        ("sample_pothole_002.jpg", generate_pothole_image, [
            "0 0.5 0.5 0.25 0.3",
            "0 0.3 0.3 0.1 0.15"
        ]),
        ("sample_crack_001.jpg", generate_crack_image, [
            "1 0.45 0.55 0.5 0.08",
            "1 0.6 0.3 0.35 0.1"
        ]),
        ("sample_crack_002.jpg", generate_crack_image, [
            "1 0.5 0.5 0.6 0.08"
        ]),
        ("sample_crazing_001.jpg", generate_crazing_image, [
            "2 0.5 0.5 0.7 0.7"
        ]),
        ("sample_crazing_002.jpg", generate_crazing_image, [
            "2 0.4 0.4 0.6 0.6",
            "2 0.7 0.7 0.4 0.4"
        ]),
        ("sample_marking_001.jpg", generate_marking_image, [
            "3 0.5 0.25 0.8 0.08",
            "3 0.5 0.75 0.7 0.08"
        ]),
        ("sample_marking_002.jpg", generate_marking_image, [
            "3 0.4 0.5 0.6 0.1"
        ]),
        ("sample_mixed_001.jpg", generate_mixed_image, [
            "0 0.2 0.4 0.15 0.2",
            "1 0.45 0.6 0.4 0.08",
            "3 0.25 0.2 0.4 0.06",
            "2 0.7 0.8 0.25 0.25"
        ])
    ]
    
    created_count = 0
    for image_name, generator_func, annotations in images_data:
        image_path = images_dir / image_name
        generator_func(str(image_path))
        create_annotations(image_path, annotations)
        created_count += 1
    
    print(f"\n✅ {created_count} images créées avec succès!")
    print(f"📍 Localisation: data/raw/images/")
    
    # Mettre à jour les métadonnées
    metadata_path = Path("data/metadata.json")
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        metadata["statistics"] = {
            "total_images": created_count,
            "total_videos": 0,
            "total_frames": created_count,
            "total_annotations": len(images_data),
            "class_distribution": {
                "pothole": 4,
                "longitudinal_crack": 3,
                "crazing": 3,
                "faded_road_marking": 3
            }
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Métadonnées mises à jour")
    
    print("\n" + "=" * 70)
    print("✅ IMAGES SYNTHÉTIQUES CRÉÉES")
    print("=" * 70)
    print(f"📸 {created_count} images générées dans data/raw/images/")
    print(f"📝 Annotations YOLO créées dans data/annotations/train/")
    print("=" * 70)

if __name__ == "__main__":
    main()
