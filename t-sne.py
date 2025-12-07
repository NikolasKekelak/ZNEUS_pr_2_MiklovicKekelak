import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from your_models_file import get_model   # <- change this

def strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_checkpoint_into_model(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    # common checkpoint formats
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
        else:
            # maybe it's already a state_dict-like dict
            state = ckpt
    else:
        raise ValueError("Checkpoint is not a dict/state_dict.")

    state = strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("Loaded checkpoint.")
    if missing:
        print("Missing keys:", missing[:10], ("..." if len(missing) > 10 else ""))
    if unexpected:
        print("Unexpected keys:", unexpected[:10], ("..." if len(unexpected) > 10 else ""))


def find_last_linear(model: nn.Module):
    last = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last = (name, m)
    if last is None:
        raise RuntimeError("No nn.Linear found in model (can't auto-pick embeddings).")
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to best_model.pth")
    ap.add_argument("--model", required=True, help="one of: jack/resnet/our-resnet/small/deep/wide")
    ap.add_argument("--data_dir", required=True, help="ImageFolder root (subfolders = classes)")
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max_points", type=int, default=3000, help="subsample for faster t-SNE")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--out", default="tsne.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # transforms (adjust to match your training!)
    tfm = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    ds = datasets.ImageFolder(args.data_dir, transform=tfm)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    model = get_model(args.model, num_classes=len(ds.classes))
    model.to(device)
    model.eval()

    load_checkpoint_into_model(model, args.ckpt, device)

    # hook to capture embeddings = input to last classifier Linear
    name, last_linear = find_last_linear(model)
    print("Using embeddings from input to last Linear:", name)

    feats = []
    labels = []

    def hook_fn(module, inp, out):
        # inp is a tuple; inp[0] is [B, D]
        feats.append(inp[0].detach().cpu())

    h = last_linear.register_forward_hook(hook_fn)

    with torch.no_grad():
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            _ = model(x)
            labels.append(y.cpu())
            if sum(t.shape[0] for t in labels) >= args.max_points:
                break

    h.remove()

    X = torch.cat(feats, dim=0).numpy()
    y = torch.cat(labels, dim=0).numpy()

    # subsample if over max_points
    if X.shape[0] > args.max_points:
        idx = np.random.choice(X.shape[0], size=args.max_points, replace=False)
        X, y = X[idx], y[idx]

    print("Embedding matrix:", X.shape)

    tsne = TSNE(
        n_components=2,
        perplexity=min(args.perplexity, max(5.0, (X.shape[0] - 1) / 3.0)),
        init="pca",
        learning_rate="auto",
        random_state=0,
    )
    Z = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=y, s=8, alpha=0.8)
    plt.title(f"t-SNE of embeddings ({args.model})")
    plt.xticks([])
    plt.yticks([])

    # optional legend (can get big if many classes)
    if len(ds.classes) <= 15:
        handles, _ = scatter.legend_elements()
        plt.legend(handles, ds.classes, title="Classes", loc="best")

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
