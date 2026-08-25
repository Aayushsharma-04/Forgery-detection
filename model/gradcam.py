import torch
from torchvision import models as models
from data.dataset import val_test_transforms
from torch import nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights = None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features,2)

model.load_state_dict(torch.load("model/artifacts/best_model_finetuned.pth",map_location = device))
model = model.to(device)
model.eval()

target_layers = [model.layer4[-1]]
image_path = "data/processed/1_Tp_D_CRD_S_O_ani10111_ani10103_10635.jpg"
pil_img = Image.open(image_path).convert("RGB")
input_tensor = val_test_transforms(pil_img).unsqueeze(0).to(device)

cam = GradCAM(model = model,target_layers = target_layers)
grayscale_cam = cam(input_tensor = input_tensor,targets = None)
grayscale_cam = grayscale_cam[0,:]

rgb_image = np.array(pil_img.resize((224,224)))/255.0
visualization = show_cam_on_image(rgb_image,grayscale_cam,use_rgb = True)
plt.imsave("model/artifacts/gradcam_example.png", visualization)
print("Saved Grad-CAM visualization to model/artifacts/gradcam_example.png")