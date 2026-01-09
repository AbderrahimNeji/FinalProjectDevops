"""
Script pour télécharger des exemples d'images et vidéos de dégradations routières
depuis des datasets publics.
"""

import os
import requests
from pathlib import Path
import zipfile
import shutil

def download_file(url, destination):
    """Télécharge un fichier depuis une URL."""
    print(f"Téléchargement: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✓ Téléchargé: {destination}")

def download_sample_images():
    """Télécharge des exemples d'images de dégradations routières."""
    
    # Créer les dossiers si nécessaire
    images_dir = Path("data/raw/images")
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📸 Téléchargement d'exemples d'images...")
    
    # Exemples d'images publiques de dégradations routières
    sample_images = [
        {
            "url": "https://raw.githubusercontent.com/sekilab/RoadDamageDetector/master/sample_images/pothole_001.jpg",
            "name": "sample_pothole_001.jpg",
            "type": "pothole"
        },
        {
            "url": "https://raw.githubusercontent.com/sekilab/RoadDamageDetector/master/sample_images/crack_001.jpg",
            "name": "sample_crack_001.jpg",
            "type": "crack"
        },
        {
            "url": "https://raw.githubusercontent.com/sekilab/RoadDamageDetector/master/sample_images/damage_001.jpg",
            "name": "sample_damage_001.jpg",
            "type": "multiple"
        }
    ]
    
    # Alternative: Utiliser des images de démonstration depuis des sources fiables
    # Si les URLs ci-dessus ne fonctionnent pas, créer des images placeholder
    
    success_count = 0
    for img_info in sample_images:
        try:
            dest = images_dir / img_info["name"]
            download_file(img_info["url"], dest)
            success_count += 1
        except Exception as e:
            print(f"⚠ Échec du téléchargement de {img_info['name']}: {e}")
    
    if success_count == 0:
        print("\n⚠ Impossible de télécharger les images depuis les sources externes.")
        print("Création d'images placeholder pour la démonstration...")
        create_placeholder_images(images_dir)
    
    return success_count

def create_placeholder_images(images_dir):
    """Crée des fichiers placeholder pour les images de démonstration."""
    
    placeholders = [
        ("sample_pothole_001.jpg.txt", "Placeholder pour image de nid-de-poule"),
        ("sample_crack_001.jpg.txt", "Placeholder pour image de fissure"),
        ("sample_crazing_001.jpg.txt", "Placeholder pour image de faïençage"),
        ("sample_marking_001.jpg.txt", "Placeholder pour image de marquage effacé")
    ]
    
    for filename, description in placeholders:
        filepath = images_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{description}\n\n")
            f.write("Pour obtenir de vraies images:\n")
            f.write("1. RDD2020: https://github.com/sekilab/RoadDamageDetector\n")
            f.write("2. Pothole Dataset: https://www.kaggle.com/datasets/chitholian/pothole-image-dataset\n")
            f.write("3. CRACK500: https://github.com/fyangneil/pavement-crack-detection\n")
        print(f"✓ Créé: {filepath}")

def download_sample_videos():
    """Télécharge ou crée des exemples de vidéos."""
    
    videos_dir = Path("data/raw/videos")
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🎥 Configuration des vidéos de démonstration...")
    
    # Créer des fichiers info pour les vidéos
    video_placeholders = [
        ("sample_road_video_001.mp4.txt", "Vidéo de route avec nids-de-poule"),
        ("sample_highway_video_001.mp4.txt", "Vidéo d'autoroute avec fissures"),
        ("sample_urban_video_001.mp4.txt", "Vidéo urbaine avec marquages effacés")
    ]
    
    for filename, description in video_placeholders:
        filepath = videos_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{description}\n\n")
            f.write("Pour obtenir de vraies vidéos:\n")
            f.write("1. Enregistrez vos propres vidéos de route (1080p, 30fps)\n")
            f.write("2. Dashcam footage datasets\n")
            f.write("3. YouTube (avec autorisation) - Recherche: 'road damage dashcam'\n\n")
            f.write("Format recommandé: MP4, H.264, 1920x1080, 30fps\n")
            f.write(f"Nom suggéré: {filename.replace('.txt', '')}\n")
        print(f"✓ Créé: {filepath}")
    
    return len(video_placeholders)

def create_sample_annotations():
    """Crée des exemples d'annotations YOLO."""
    
    annotations_dir = Path("data/annotations/train")
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📝 Création d'exemples d'annotations...")
    
    # Annotations correspondant aux images placeholder
    sample_annotations = {
        "sample_pothole_001.txt": [
            "0 0.5 0.5 0.3 0.4  # Pothole au centre",
            "0 0.2 0.3 0.15 0.2  # Petit pothole en haut à gauche"
        ],
        "sample_crack_001.txt": [
            "1 0.4 0.6 0.6 0.05  # Fissure longitudinale",
            "1 0.7 0.3 0.4 0.08  # Fissure diagonale"
        ],
        "sample_crazing_001.txt": [
            "2 0.5 0.5 0.8 0.8  # Faïençage large zone"
        ],
        "sample_marking_001.txt": [
            "3 0.5 0.8 0.9 0.1  # Marquage routier effacé"
        ]
    }
    
    for filename, annotations in sample_annotations.items():
        filepath = annotations_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# Format YOLO: class x_center y_center width height (normalized 0-1)\n")
            for annotation in annotations:
                f.write(f"{annotation}\n")
        print(f"✓ Créé: {filepath}")
    
    return len(sample_annotations)

def create_sample_gps_data():
    """Crée des données GPS pour les exemples."""
    
    gps_dir = Path("data/gps")
    gps_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n🗺️  Création de données GPS d'exemple...")
    
    import json
    
    # GPS data pour Paris (exemple)
    gps_data = [
        {
            "frame_id": "sample_pothole_001_frame_000",
            "timestamp": "2026-01-09T10:15:30Z",
            "latitude": 48.8566,
            "longitude": 2.3522,
            "altitude": 35.0,
            "accuracy": 5.2,
            "detections": [
                {"class": "pothole", "confidence": 0.89, "bbox": [0.5, 0.5, 0.3, 0.4]},
                {"class": "pothole", "confidence": 0.76, "bbox": [0.2, 0.3, 0.15, 0.2]}
            ]
        },
        {
            "frame_id": "sample_crack_001_frame_000",
            "timestamp": "2026-01-09T10:16:45Z",
            "latitude": 48.8575,
            "longitude": 2.3530,
            "altitude": 36.0,
            "accuracy": 4.8,
            "detections": [
                {"class": "longitudinal_crack", "confidence": 0.92, "bbox": [0.4, 0.6, 0.6, 0.05]},
                {"class": "longitudinal_crack", "confidence": 0.85, "bbox": [0.7, 0.3, 0.4, 0.08]}
            ]
        },
        {
            "frame_id": "sample_crazing_001_frame_000",
            "timestamp": "2026-01-09T10:18:20Z",
            "latitude": 48.8584,
            "longitude": 2.3538,
            "altitude": 37.0,
            "accuracy": 5.5,
            "detections": [
                {"class": "crazing", "confidence": 0.88, "bbox": [0.5, 0.5, 0.8, 0.8]}
            ]
        },
        {
            "frame_id": "sample_marking_001_frame_000",
            "timestamp": "2026-01-09T10:19:55Z",
            "latitude": 48.8593,
            "longitude": 2.3546,
            "altitude": 38.0,
            "accuracy": 4.9,
            "detections": [
                {"class": "faded_road_marking", "confidence": 0.81, "bbox": [0.5, 0.8, 0.9, 0.1]}
            ]
        }
    ]
    
    filepath = gps_dir / "sample_detections_gps.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(gps_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Créé: {filepath}")
    return 1

def update_metadata():
    """Met à jour les métadonnées avec les statistiques des exemples."""
    
    import json
    
    metadata_path = Path("data/metadata.json")
    
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Mettre à jour les statistiques avec les exemples
        metadata["statistics"] = {
            "total_images": 4,
            "total_videos": 3,
            "total_frames": 4,
            "total_annotations": 6,
            "class_distribution": {
                "pothole": 2,
                "longitudinal_crack": 2,
                "crazing": 1,
                "faded_road_marking": 1
            }
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Métadonnées mises à jour: {metadata_path}")

def main():
    """Fonction principale."""
    
    print("=" * 60)
    print("🚗 Téléchargement de données d'exemple pour détection de dégradations routières")
    print("=" * 60)
    
    # Télécharger/créer les images
    images_count = download_sample_images()
    
    # Télécharger/créer les vidéos
    videos_count = download_sample_videos()
    
    # Créer les annotations
    annotations_count = create_sample_annotations()
    
    # Créer les données GPS
    gps_count = create_sample_gps_data()
    
    # Mettre à jour les métadonnées
    update_metadata()
    
    print("\n" + "=" * 60)
    print("✅ DONNÉES D'EXEMPLE CRÉÉES")
    print("=" * 60)
    print(f"📸 Images: {images_count} fichiers dans data/raw/images/")
    print(f"🎥 Vidéos: {videos_count} fichiers info dans data/raw/videos/")
    print(f"📝 Annotations: {annotations_count} fichiers dans data/annotations/train/")
    print(f"🗺️  GPS: {gps_count} fichier dans data/gps/")
    print("\n⚠️  Note: Certains fichiers sont des placeholders.")
    print("   Consultez les README pour obtenir de vraies données:")
    print("   - data/raw/images/README.md")
    print("   - data/raw/videos/README.md")
    print("   - data/README.md")
    print("\n💡 Datasets recommandés:")
    print("   - RDD2020: https://github.com/sekilab/RoadDamageDetector")
    print("   - Pothole Dataset: https://www.kaggle.com/datasets/chitholian/pothole-image-dataset")
    print("   - CRACK500: https://github.com/fyangneil/pavement-crack-detection")
    print("=" * 60)

if __name__ == "__main__":
    main()
