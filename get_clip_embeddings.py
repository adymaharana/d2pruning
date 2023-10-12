import os
import clip
import torch
import numpy as np
from tqdm import tqdm
from torchvision.datasets import CIFAR100
from torchvision import datasets, transforms

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

mean = [0.507, 0.487, 0.441]
std = [0.267, 0.256, 0.276]
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("../data/cifar100"), download=True, train=True, transform=preprocess)
print(len(cifar100))
trainloader = torch.utils.data.DataLoader(
    cifar100, batch_size=16, shuffle=False, num_workers=16, drop_last=False)

# Calculate features
all_image_features = None
with torch.no_grad():
    for idx, batch in tqdm(enumerate(trainloader)):
        # Prepare the inputs
        image, targets = batch
        image_features = model.encode_image(image.to(device)).detach().cpu().numpy()

        if all_image_features is None:
            all_image_features = image_features
        else:
            all_image_features = np.concatenate((all_image_features, image_features), axis=0)

print(all_image_features.shape)
np.save('clip_cifar100.npy', all_image_features)

# Pick the top 5 most similar labels for the image
