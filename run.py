#!/usr/bin/env python
"""
Toy Example for EBH AUROC Evaluation:
Uses CIFAR10 or CIFAR100 as the in-distribution (ID) dataset (with a custom pretrained checkpoint)
and an out-of-distribution (OOD) dataset (e.g., CIFAR10, CIFAR100, SVHN, MNIST, or FashionMNIST)
with a ResNet18 to compute and plot per-layer AUROC scores.
"""

import argparse
import torch

from utils import plot_auroc, compute_auroc, run_inference, register_hooks, get_model, get_dl


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the ID and OOD datasets and their respective DataLoaders.
    id_loader, id_name, ood_loader, ood_name = get_dl(args)

    # Initialize a ResNet18 model, load the pretrained checkpoint.
    model = get_model(device, id_name)

    # Register forward hooks on all Conv2d, Linear, and BatchNorm2d layers.
    layer_order = register_hooks(model)

    # Run inference on the ID and OOD datasets.
    id_labels, id_scores, ood_labels, ood_scores = run_inference(args, device, id_loader, id_name, model, ood_loader)

    # Compute AUROC per hooked layer.
    auroc_per_layer = compute_auroc(id_labels, id_scores, layer_order, ood_labels, ood_scores)

    # Plot the per-layer AUROC scores.
    plot_auroc(args, auroc_per_layer, id_name, layer_order, ood_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Toy Example: EBH AUROC Evaluation with CIFAR10/CIFAR100 as ID and a torchvision OOD dataset. "
                    "ID must be 'cifar10' or 'cifar100' (using custom checkpoints).")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for DataLoader")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory to load datasets (for CIFAR, the dataset will be downloaded here; "
                             "for checkpoints, ensure the proper structure exists)")
    parser.add_argument("--output_dir", type=str, default="./figures",
                        help="Directory to save plots")
    parser.add_argument("--id_dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"],
                        help="Name of the in-distribution dataset (must be 'cifar10' or 'cifar100')")
    parser.add_argument("--ood_dataset", type=str, default="svhn", choices=["cifar10", "cifar100", "svhn", "mnist", "fashionmnist"],
                        help="Name of the out-of-distribution dataset")
    args = parser.parse_args()
    main(args)
