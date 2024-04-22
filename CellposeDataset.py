#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 01:56:57 2024

@author: ian
"""
import os
import torch
from PIL import Image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import ToTensor
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import numpy as np
from torchvision.ops import nms
import utils
import torchvision
from engine import train_one_epoch, evaluate

label_path_train = image_path_train = "data/train"
label_path_test = image_path_test = "data/test"
images = [
    image
    for image in sorted(os.listdir(os.path.join(image_path_train)))
    if image.endswith("img.png")
]
labels = [
    image
    for image in sorted(os.listdir(os.path.join(label_path_train)))
    if image.endswith("masks.png")
]

assert len(images) == len(labels)


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def normalize_img(tensor_im):
    normalized_tensor = (tensor_im - tensor_im.min()) / (
        tensor_im.max() - tensor_im.min()
    )
    uint8_tensor = (normalized_tensor * 255).clamp(0, 255).to(torch.uint8)
    return uint8_tensor


def seg_to_mask(seg):
    """
    Transforms segmentations to a mask representation
    """
    # We get the unique colors, as these would be the object ids.
    obj_ids = torch.unique(seg)

    # first id is the background, so remove it.
    obj_ids = obj_ids[1:]

    # split the color-encoded mask into a set of boolean masks.
    # Note that this snippet would work as well if the masks were float values instead of ints.
    masks = seg == obj_ids[:, None, None]
    return masks


class CellposeDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.img_names = [
            image
            for image in sorted(os.listdir(root))
            if image.endswith("img.png")
        ]
        self.mask_names = [
            mask
            for mask in sorted(os.listdir(root))
            if mask.endswith("masks.png")
        ]

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.img_names[idx])
        seg_path = os.path.join(self.root, self.mask_names[idx])
        im = Image.open(img_path)
        convert_tensor = ToTensor()
        tensor_im = convert_tensor(im)
        img = normalize_img(tensor_im)
        labels = Image.open(seg_path)
        labs = convert_tensor(labels)

        masks = seg_to_mask(labs)
        num_objs = len(masks)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        if any(area.cpu().numpy() == 0):
            print(f"problem in {idx}")
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(
            boxes, format="XYXY", canvas_size=F.get_size(img)
        )
        #        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.img_names)


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


ds_train = CellposeDataset(label_path_train, get_transform(train=True))
idx = 21
drawn_boxes = draw_bounding_boxes(
    normalize_img(ds_train.__getitem__(idx)[0]),
    ds_train.__getitem__(idx)[1]["boxes"],
    colors="red",
)
show(drawn_boxes)

device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# our dataset has two classes only - background and person
num_classes = 2
# use our dataset and defined transformations
ds_test = CellposeDataset(label_path_test, get_transform(train=False))

# %%
# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    ds_train,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    collate_fn=utils.collate_fn,
)

data_loader_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    collate_fn=utils.collate_fn,
)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights="DEFAULT"
    )
    return model


# get the model using our helper function
model = get_model_instance_segmentation(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params, lr=0.005, momentum=0.9, weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=3, gamma=0.1
)

# let's train it just for 2 epochs
num_epochs = 5

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(
        model, optimizer, data_loader_test, device, epoch, print_freq=10
    )
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
#    evaluate(model, data_loader_test, device=device)

print("That's it!")

# %% Testing

img_path = "data/test/010_img.png"
eval_transform = get_transform(train=False)
im = Image.open(img_path)
convert_tensor = ToTensor()
tensor_im = convert_tensor(im)

model.eval()
with torch.no_grad():
    x = eval_transform(tensor_im)
    # convert RGBA -> RGB and move to device
    x = x[:3, ...].to(device)
    predictions = model([x])
    pred = predictions[0]

tensor_im = normalize_img(tensor_im)
tensor_im = tensor_im[:3, ...]
idx_after = nms(pred["boxes"], pred["scores"], iou_threshold=0.2)
pred_labels = [
    f"cell: {score:.3f}"
    for label, score in zip(
        pred["labels"][idx_after], pred["scores"][idx_after]
    )
]
pred_boxes = pred["boxes"][idx_after].long()
output_image = draw_bounding_boxes(
    tensor_im, pred_boxes, pred_labels, colors="red"
)

plt.figure(figsize=(12, 12))
plt.imshow(output_image.permute(1, 2, 0))
