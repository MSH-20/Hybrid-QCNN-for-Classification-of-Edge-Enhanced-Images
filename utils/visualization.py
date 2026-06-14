import torch
import matplotlib.pyplot as plt


def show_predictions(model, loader, n_show=8):
    model.eval()
    images, labels = next(iter(loader))
    n_show = min(n_show, len(images))
    with torch.no_grad():
        preds = model(images).argmax(1)

    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 2, 4))
    for i in range(n_show):
        axes[0, i].imshow(images[i].squeeze(), cmap='gray')
        axes[0, i].set_title(f"True: {labels[i].item()}")
        axes[0, i].axis('off')

        axes[1, i].imshow(images[i].squeeze(), cmap='gray')
        axes[1, i].set_title(f"Pred: {preds[i].item()}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()
