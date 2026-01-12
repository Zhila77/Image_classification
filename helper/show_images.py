import torch 
import torch.nn as nn
import torchvision 
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms



def show_images(images, num_samples=20, cols =4):
    plt.figure(figsize=(15, 15))
    idx = int(len(images) / (num_samples))
    print(images)

    for i, img in enumerate(images):
        if i % idx ==0:
            plt.subplot(int(num_samples/cols) + 1,cols, int(i/idx) + 1)
            plt.imshow(to_pil_image(img[0]))
    plt.show()


dataset = OxfordIIITPet(root='./data', download=True, transform=transforms.Compose([transforms.Resize((144, 144)), transforms.ToTensor()]) )
show_images(dataset)