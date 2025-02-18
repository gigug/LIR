import os
from collections import OrderedDict

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from torch import nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, MNIST, FashionMNIST
from torchvision.models import resnet18

activation_scores = OrderedDict()


def get_hook(layer_name):
    """
    Returns a hook function that computes the energy
    for each sample and stores them in a global OrderedDict.
    """
    def hook(module, input, output):
        # Compute the energy for each sample in the batch.
        act = output.view(output.shape[0], -1)
        scores = -torch.logsumexp(act, dim=1).detach().cpu()
        if layer_name not in activation_scores:
            activation_scores[layer_name] = []
        activation_scores[layer_name].extend(scores)
    return hook


def run_inference_with_labels(model, dataloader, device, label_value):
    """
    Runs the model over the dataloader.
    Collects activation scores (via hooks) and stores ground-truth labels.
    """
    global activation_scores
    activation_scores = OrderedDict()
    collected_labels = OrderedDict()
    model.eval()

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            batch_size = images.size(0)
            _ = model(images)
            for layer in activation_scores:
                if layer not in collected_labels:
                    collected_labels[layer] = []
                collected_labels[layer].extend([label_value] * batch_size)

    return activation_scores.copy(), collected_labels.copy()


def get_transform(id_name, ood_name):
    """
    Returns the appropriate transform for CIFAR10/CIFAR100.
    We use CIFAR normalization.
    """
    id_name = id_name.lower()
    if id_name == "cifar10":
        # Standard CIFAR10 normalization
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
    elif id_name == "cifar100":
        # Standard CIFAR100 normalization
        mean = [0.5071, 0.4865, 0.4409]
        std = [0.2673, 0.2564, 0.2761]
    else:
        raise ValueError("ID dataset must be 'cifar10' or 'cifar100'")

    transforms_id = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32)),
        transforms.Normalize(mean=mean, std=std)
    ])

    if ood_name in ['mnist', 'fashionmnist']:
        transforms_ood = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Resize((32, 32)),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        transforms_ood = transforms_id
    return transforms_id, transforms_ood


def load_dataset(name, root, transform):
    """
    Loads a dataset given its name.
    """
    name = name.lower()
    if name == "cifar10":
        return CIFAR10(root=root, train=False, download=True, transform=transform)
    elif name == "cifar100":
        return CIFAR100(root=root, train=False, download=True, transform=transform)
    elif name == "svhn":
        return SVHN(root=root, split='test', download=True, transform=transform)
    elif name == "mnist":
        return MNIST(root=root, train=False, download=True, transform=transform)
    elif name == "fashionmnist":
        return FashionMNIST(root=root, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {name}")


def plot_auroc(args, auroc_per_layer, id_name, layer_order, ood_name):
    # Determine color for each bar:
    # Use 'orange' for the final fc layer and color intermediate layers green if their AUROC > fc AUROC.
    fc_auroc = auroc_per_layer.get("fc", None)
    colors = []
    for layer in layer_order:
        if layer == "fc":
            colors.append("orange")
        elif fc_auroc is not None and auroc_per_layer.get(layer, 0) > fc_auroc:
            colors.append("green")
        else:
            colors.append("orange")

    # Plot the per-layer AUROC scores.
    layers = list(auroc_per_layer.keys())
    auroc_vals = list(auroc_per_layer.values())
    x = np.arange(len(layers))
    width = 0.6
    fig, ax = plt.subplots(figsize=(12, 6))
    rects = ax.bar(x, auroc_vals, width, color=colors)

    # Plot a horizontal dotted line at the fc layer AUROC.
    if fc_auroc is not None:
        ax.axhline(y=fc_auroc, color="red", linestyle="dotted", label="fc AUROC")
    ax.set_ylabel('AUROC')
    ax.set_title(f'Per-layer AUROC Scores: {id_name} (ID) vs. {args.ood_dataset.lower()} (OOD)')
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha='right', fontsize=8)
    ax.set_ylim([0.5, 1])
    ax.legend()
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    plot_path = os.path.join(args.output_dir, f"auroc_scores_{id_name}_{ood_name}.png")
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")
    plt.show()


def compute_auroc(id_labels, id_scores, layer_order, ood_labels, ood_scores):
    """
    Computes the AUROC scores for each layer.
    """
    auroc_per_layer = OrderedDict()
    for layer in layer_order:
        id_vals = id_scores.get(layer, [])
        ood_vals = ood_scores.get(layer, [])
        scores = np.array(id_vals + ood_vals)
        id_lbls = id_labels.get(layer, [])
        ood_lbls = ood_labels.get(layer, [])
        labels = np.array(id_lbls + ood_lbls)

        try:
            auroc = roc_auc_score(labels, scores)
            # Adjust if AUROC is below 0.5 (i.e., energies for ID and OOD are inversely correlated on that layer).
            if auroc < 0.5:
                auroc = 1 - auroc
        except Exception:
            auroc = float('nan')
        auroc_per_layer[layer] = auroc
    # Print per-layer and overall AUROC scores.
    print("\nPer-layer AUROC scores (ID vs. OOD) in network order:")
    for layer, score in auroc_per_layer.items():
        print(f"  {layer:40s}: AUROC = {score:.4f}")
    return auroc_per_layer


def run_inference(args, device, id_loader, id_name, model, ood_loader):
    """
    Run inference on the ID and OOD datasets.
    """
    print(f"Running inference on ID dataset ({id_name}) ...")
    id_scores, id_labels = run_inference_with_labels(model, id_loader, device, label_value=0)
    print(f"Running inference on OOD dataset ({args.ood_dataset.lower()}) ...")
    ood_scores, ood_labels = run_inference_with_labels(model, ood_loader, device, label_value=1)
    return id_labels, id_scores, ood_labels, ood_scores


def register_hooks(model):
    """
    Register forward hooks for each Conv2d, Linear, and BatchNorm2d layer in the model.
    """
    hook_handles = []
    layer_order = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            handle = module.register_forward_hook(get_hook(name))
            hook_handles.append(handle)
            layer_order.append(name)  # Store the layer order as they appear in model
    return layer_order


def get_model(device, id_name):
    """
    Load a pretrained ResNet18 model for CIFAR10 or CIFAR100.
    """
    model = resnet18(num_classes=10 if id_name == "cifar10" else 100)
    model = model.to(device)
    checkpoint_path = None
    if id_name == "cifar10":
        checkpoint_path = os.path.join("checkpoints", "resnet18_cifar10.pth")
    elif id_name == "cifar100":
        checkpoint_path = os.path.join("checkpoints", "resnet18_cifar100.pth")
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    return model


def get_dl(args):
    """
    Load the ID and OOD datasets and return their respective DataLoaders.
    """
    id_name = args.id_dataset.lower()
    ood_name = args.ood_dataset.lower()
    if id_name not in ["cifar10", "cifar100"]:
        raise ValueError("ID dataset must be either 'cifar10' or 'cifar100'")
    transform_id, transform_ood = get_transform(id_name, ood_name)
    id_dataset = load_dataset(id_name, args.data_dir, transform_id)
    ood_dataset = load_dataset(args.ood_dataset.lower(), args.data_dir, transform_ood)
    id_loader = DataLoader(id_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    ood_loader = DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    return id_loader, id_name, ood_loader, ood_name
