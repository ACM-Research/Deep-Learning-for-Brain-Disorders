from fastai.vision.all import *
import torch
import PIL

def is_yes(x):
    return "y" in x.lower()

def main():
    """
    path = "./data"
    dls = ImageDataLoaders.from_name_func(
        path, get_image_files(path), valid_pct=0.2, seed=42,
        label_func=is_yes, item_tfms=Resize(224))
    learn = cnn_learner(dls, resnet50, metrics=accuracy)
    learn.fine_tune(5)
    """

    # VALIDATION
    learn = torch.load("brain_tumor")
    img = PILImage.create("./data/N17.jpg")
    is_cancer, label, probs = learn.predict(img)
    print(f"Is this cancer? {is_cancer}")
    print(f"label: {label}")
    print(f"Probability it's cancer: {probs[1].item():.6f}")

if __name__ == "__main__":
    main()
