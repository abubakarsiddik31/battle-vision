"""
Dataset preparation script for KIIT-MiTA multi-label classification.
Parses YOLO format labels and creates multi-hot encoded labels.
"""

import os
from pathlib import Path
from collections import defaultdict
import json
from typing import Dict, List, Set

# Class names from the YAML config
CLASS_NAMES = [
    "Artilary",
    "Missile",
    "Radar",
    "M. Rocket Launcher",
    "Soldier",
    "Tank",
    "Vehicle"
]

# Dataset root
DATASET_ROOT = Path("/home/abubakar/Desktop/Research/DL-assignment/KIIT-MiTA")


def parse_yolo_label(label_path: Path) -> List[int]:
    """Parse YOLO format label file and return list of class IDs present."""
    if not label_path.exists():
        return []

    class_ids = set()
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if parts:
                    class_id = int(parts[0])
                    class_ids.add(class_id)

    return sorted(class_ids)


def create_multihot_labels(class_ids: List[int], num_classes: int = 7) -> List[int]:
    """Create multi-hot encoded label vector."""
    label = [0] * num_classes
    for class_id in class_ids:
        if 0 <= class_id < num_classes:
            label[class_id] = 1
    return label


def process_split(split_name: str) -> Dict:
    """Process a data split (train, test, valid) and return annotations."""
    split_path = DATASET_ROOT / split_name
    images_dir = split_path / "images"
    labels_dir = split_path / "labels"

    annotations = []
    class_counts = defaultdict(int)
    num_objects_per_image = []

    for image_path in images_dir.glob("*.jpeg"):
        # Get corresponding label file
        label_name = image_path.stem + ".txt"
        label_path = labels_dir / label_name

        # Parse label to get class IDs
        class_ids = parse_yolo_label(label_path)
        multihot_label = create_multihot_labels(class_ids)

        # Track statistics
        num_objects_per_image.append(len(class_ids))
        for class_id in class_ids:
            class_counts[class_id] += 1

        annotations.append({
            "image_path": str(image_path.relative_to(DATASET_ROOT)),
            "label": multihot_label,
            "class_ids": class_ids
        })

    # Compute statistics
    stats = {
        "split": split_name,
        "num_images": len(annotations),
        "class_distribution": {CLASS_NAMES[i]: class_counts[i] for i in range(len(CLASS_NAMES))},
        "avg_objects_per_image": sum(num_objects_per_image) / len(num_objects_per_image) if num_objects_per_image else 0,
        "max_objects_per_image": max(num_objects_per_image) if num_objects_per_image else 0,
    }

    return {"annotations": annotations, "stats": stats}


def main():
    """Main function to prepare the dataset."""
    print("=" * 60)
    print("KIIT-MiTA Dataset Preparation for Multi-Label Classification")
    print("=" * 60)

    # Process all splits
    splits = ["train", "test", "valid"]
    all_data = {}

    for split in splits:
        print(f"\nProcessing {split} split...")
        data = process_split(split)
        all_data[split] = data

        # Print statistics
        stats = data["stats"]
        print(f"  Images: {stats['num_images']}")
        print(f"  Avg objects/image: {stats['avg_objects_per_image']:.2f}")
        print(f"  Max objects/image: {stats['max_objects_per_image']}")
        print(f"  Class distribution:")
        for class_name, count in stats['class_distribution'].items():
            print(f"    - {class_name}: {count}")

    # Save annotations to JSON files
    output_dir = Path("/home/abubakar/Desktop/Research/DL-assignment/data")
    output_dir.mkdir(exist_ok=True)

    for split in splits:
        output_path = output_dir / f"{split}_annotations.json"
        with open(output_path, 'w') as f:
            json.dump(all_data[split]["annotations"], f, indent=2)
        print(f"\nSaved {split} annotations to {output_path}")

    # Save metadata
    metadata = {
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "splits": {
            split: all_data[split]["stats"]
            for split in splits
        }
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
