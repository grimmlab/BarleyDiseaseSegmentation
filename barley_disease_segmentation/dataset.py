"""
Dataset class for barley leaf disease segmentation.

Handles loading of leaf scan images and corresponding masks for:
- Brown rust (binary)
- Ramularia leaf spot (binary) 
- Multiclass segmentation (both diseases)
"""

import os
import re
from collections import Counter
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage.color import rgb2gray
from torch.utils.data import Dataset

from barley_disease_segmentation.config import PACKAGE_DIR

__all__ = ['BarleyLeafDataset']

class BarleyLeafDataset(Dataset):
    def __init__(
            self,
            all_genotypes_dir,
            genotypes_list,
            task="multiclass",
            augmentations=None,
            standardize=True,
            exclude_invalid=True,
            config_path=PACKAGE_DIR / "config.py",
            calculate_weights=True
    ):
        """
        Dataset for barley leaf disease segmentation.

        Args:
            all_genotypes_dir: Path containing processed genotype folders
            genotypes_list: List of genotype IDs to include
            task: "brownrust", "ramularia", or "multiclass"
            augmentations: Albumentations pipeline for data augmentation
            standardize: Whether to normalize images
            exclude_invalid: Exclude all-white/all-black patches
            config_path: Path to config file
            calculate_weights: Compute class weights from dataset
        """
        self.all_genotypes_dir = all_genotypes_dir
        self.genotypes_list = genotypes_list
        self.augmentations = augmentations
        self.standardize = standardize
        self.exclude_invalid = exclude_invalid
        self.CONFIG_PATH = config_path
        self.task = task

        self.augmentations = self._initialize_augmentations(augmentations)
        self.data = {}  # nested structure
        self.patches = self._initialize_patches()  # flattened structure

        if self.task in ["ramularia", "brownrust"]:
            self.num_classes = 2
        else:
            self.num_classes = self._load_or_compute_num_classes()

        self.mean = None
        self.std = None
        if calculate_weights:
            self.class_weights = self.calculate_class_weights()

    def _initialize_patches(self):
        """
        Build flat list of patches from genotype folders.

        Returns:
            List of patch dictionaries with image/mask paths and metadata.
        """
        patches = []
        for genotype in self.genotypes_list:
            genotype_data = {}
            image_dir = os.path.join(
                self.all_genotypes_dir,
                f"{genotype}/cropped_leaves_{genotype}_boxed_preprocessed_patches"
            )
            mask_dir = os.path.join(
                self.all_genotypes_dir,
                f"{genotype}/annotated_masks_{genotype}_boxed_preprocessed_patches"
            )

            if not os.path.exists(image_dir):
                print(f"Image directory not found: {image_dir}")

            for root, _, files in os.walk(image_dir):
                for file in sorted(files):
                    match = re.search(r"_(\d+)_patch_(\d+)_y(-?\d+)_x(-?\d+)_size(\d+)", file)
                    if not match:
                        continue

                    id_leaf, id_patch, y, x, patch_size = match.groups()
                    mask_path = os.path.join(
                        mask_dir,
                        f"annotated_mask_{id_leaf}_patch_{id_patch}_y{y}_x{x}_size{patch_size}.tiff"
                    )

                    if not os.path.exists(mask_path):
                        continue

                    if self.exclude_invalid:
                        image_array = np.array(rgb2gray(Image.open(os.path.join(root, file))))
                        if image_array.max() == 0 or image_array.min() == 255:
                            continue

                    img_name = f"{genotype}_{id_leaf.zfill(2)}_patch_{id_patch.zfill(2)}_y{y}_x{x}_size{patch_size}"
                    patches.append({
                        'img_path': os.path.join(root, file),
                        'mask_path': mask_path,
                        'img_name': img_name,
                        'x_offset': int(x),
                        'y_offset': int(y),
                        'orig_image_id': f"{genotype}_{id_leaf.zfill(2)}"
                    })

                    if id_leaf not in genotype_data:
                        genotype_data[id_leaf] = {'img': [], 'mask': [], 'img_name': []}
                    genotype_data[id_leaf]['img'].append(os.path.join(root, file))
                    genotype_data[id_leaf]['mask'].append(mask_path)
                    genotype_data[id_leaf]['img_name'].append(img_name)

            self.data[genotype] = genotype_data

        return patches

    def __len__(self):
        """Return number of patches."""
        return len(self.patches)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Returns:
            tuple: (image, mask, metadata, background_mask)
        """
        patch = self.patches[idx]
        img_path = patch['img_path']
        mask_path = patch['mask_path']

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        background_mask = np.all(image == 255, axis=-1)
        background_mask = torch.tensor(background_mask, dtype=torch.bool)

        # Task-specific mask processing
        if self.task == "ramularia":
            mask[mask == 1] = 0  # remove brown rust
            mask[mask == 2] = 1  # convert ramularia to 1
        elif self.task == "brownrust":
            mask[mask == 2] = 0  # remove ramularia

        if self.num_classes > 2:
            mask = mask.astype(np.uint8)
        else:
            mask = (mask > 0).astype(np.uint8)

        # Apply augmentations
        if self.augmentations:
            augmented = self.augmentations(image=image, mask=mask, background_mask=background_mask)
            image = augmented["image"]
            mask = augmented["mask"]
            background_mask = augmented["background_mask"]

        image = transforms.ToTensor()(image)

        if self.standardize and self.mean and self.std:
            image = transforms.Normalize(self.mean, self.std)(image)

        mask = torch.tensor(mask, dtype=torch.long)
        metadata = {
            "x_offset": patch["x_offset"],
            "y_offset": patch["y_offset"],
            "img_name": patch["img_name"],
            "orig_image_id": patch["orig_image_id"]
        }

        return image, mask.long(), metadata, background_mask.bool()

    def _initialize_augmentations(self, augmentations):
        """Validate and return augmentation pipeline."""
        if callable(augmentations):
            return augmentations
        elif augmentations is None:
            return None
        else:
            raise ValueError("augmentations must be callable or None")

    def compute_mean_and_std(self, dataloader, type='imagenet'):
        """
        Compute image statistics for normalization.

        Args:
            dataloader: DataLoader for computing statistics
            type: 'imagenet' (use pretrained stats) or 'internal' (compute from data)

        Returns:
            tuple: (mean, std) per channel
        """
        if type == 'imagenet':
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
            return self.mean, self.std

        mean_accumulator = torch.zeros(3)
        std_accumulator = torch.zeros(3)
        num_pixels = 0

        for images, _, _ in dataloader:
            batch_size, C, H, W = images.shape
            mean_accumulator += images.mean(dim=[0, 2, 3]) * batch_size
            std_accumulator += images.std(dim=[0, 2, 3]) * batch_size
            num_pixels += batch_size

        mean = (mean_accumulator / num_pixels).tolist()
        std = (std_accumulator / num_pixels).tolist()

        self.mean = mean
        self.std = std
        return mean, std

    def _load_or_compute_num_classes(self):
        """Get number of classes from config or compute from masks."""
        num_classes = None

        if os.path.exists(self.CONFIG_PATH):
            with open(self.CONFIG_PATH, "r") as f:
                config_text = f.read()

            try:
                namespace = {}
                exec(config_text, {}, namespace)
                if "NUM_CLASSES" in namespace:
                    num_classes = namespace["NUM_CLASSES"]
            except Exception:
                pass

        if num_classes is None:
            num_classes = self._compute_num_classes()
            self._update_config_line(self.CONFIG_PATH, 'NUM_CLASSES', num_classes)

        return num_classes

    def _compute_num_classes(self):
        """Scan masks to count unique labels."""
        unique_labels = set()
        mask_paths = [patch['mask_path'] for patch in self.patches if 'mask_path' in patch]

        for mask_path in mask_paths:
            mask = np.array(Image.open(mask_path))
            unique_labels.update(np.unique(mask))

        return len(unique_labels)

    def calculate_class_weights(self):
        """Compute class weights based on pixel frequencies."""
        print(f"Calculating class weights for {self.task} dataset...")
        class_pixel_counts = Counter()

        for i in range(len(self)):
            _, mask, _, _ = self[i]
            mask_np = mask.numpy()
            unique, counts = np.unique(mask_np, return_counts=True)
            class_pixel_counts.update(dict(zip(unique, counts)))

        print(f"Class distribution: {dict(class_pixel_counts)}")

        if not class_pixel_counts:
            return torch.tensor([1.0], dtype=torch.float32)

        num_classes = max(class_pixel_counts.keys()) + 1
        print(f"Dataset contains {num_classes} classes")

        total_pixels = sum(class_pixel_counts.values())
        class_weights = []

        for cls in range(num_classes):
            count = class_pixel_counts.get(cls, 1)
            weight = total_pixels / (count * num_classes)
            class_weights.append(weight)

        weights = torch.tensor(class_weights, dtype=torch.float32)
        weights = weights / weights.sum() * num_classes

        print(f"Final class weights: {weights.tolist()}")
        return weights

    def _update_config_line(self, config_path, var_name, value_str):
        """
        Update a variable in config.py.

        Args:
            config_path: Path to config file
            var_name: Variable name to update
            value_str: Value to write
        """
        lines = []
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                lines = f.readlines()

        updated = False
        for i, line in enumerate(lines):
            if line.strip().startswith(f"{var_name} ="):
                lines[i] = f"{var_name} = {value_str}\n"
                updated = True
                break

        if not updated:
            if not lines or lines[-1].endswith("\n"):
                lines.append(f"{var_name} = {value_str}\n")
            else:
                lines.append(f"\n{var_name} = {value_str}\n")

        with open(config_path, "w") as f:
            f.writelines(lines)

        print(f"Updated {var_name} in {config_path}")