from fastai.vision.all import *
import torch

lbl_dict = dict(
    MildDemented = "Mild",
    ModerateDemented = "Moderate",
    NonDemented = "Normal",
    VeryMildDemented = "VeryMild"
)

def label_func(fname):
    return lbl_dict[parent_label(fname)]

def main():
    """
    dblock = DataBlock(blocks = (ImageBlock, CategoryBlock),
                       get_items = get_image_files,
                       get_y = label_func,
                       item_tfms = RandomResizedCrop(128, min_scale=0.35),
                       batch_tfms = Normalize.from_stats(*imagenet_stats),
                       splitter = GrandparentSplitter(valid_name="train"))
    dls = dblock.dataloaders("data")
    learn = cnn_learner(dls, resnet50, metrics=accuracy)
    learn.fine_tune(5)
    """

    # VALIDATION
    learn = torch.load("alzheimers")
    img = PILImage.create("../../../MAIN/src/images/AlsoDemented.jpg")
    label, what, probs = learn.predict(img)
    print(f"Label is: {label}")
    print(f"Probability it's alzheimers: {probs}")
    print(f"what is this {what}")
    # torch.save(learn, "alzheimers")

if __name__ == "__main__":
    main()
