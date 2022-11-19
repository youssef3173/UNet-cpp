# Predict Labels for an Image:
from unet_model import UNet

import torch
import torchvision
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image

import matplotlib.pyplot as plt
from PIL import Image
from random import randint
import cv2
import os
import uuid
import time


t1 = time.time()


chkpt_path = "./Unet.pt"

n_channels, n_classes = 3, 8
net = UNet( n_channels, n_classes)

if os.path.isfile( chkpt_path):
    net.load_state_dict( torch.load(chkpt_path, map_location=torch.device('cpu') ))


# our transform that is applied to all incoming images
transform_image = torchvision.transforms.Compose([
    lambda x: Image.open( x),
    torchvision.transforms.Resize(size=(128, 256), interpolation=InterpolationMode.BILINEAR),
    torchvision.transforms.ToTensor(),
    lambda x: x.unsqueeze(0)
])


datadir = "./Cityscapes/leftImg8bit/val/frankfurt/" 
files = os.listdir( datadir)
idx = randint( 0, len(files)-1)
image = transform_image( datadir + files[idx])

datadir = "./Cityscapes/gtFine/val/frankfurt/" 
color = transform_image( datadir + files[idx][:-15]+"gtFine_color.png")

labels = net.forward( image)


pred_class = torch.zeros((labels.size()[0], labels.size()[2], labels.size()[3]))
for idx in range(0, labels.size()[0]):
    pred_class[idx] = torch.argmax( labels[idx], dim=0).cpu().int()
pred_class = pred_class.unsqueeze(1).float()


# # debug saving generated classes to file
if not os.path.isdir( "./Results"):
	os.mkdir( "./Results")


p = (pred_class.float()/labels.size()[1])[0]  # prediction
p = p.expand(3, p.size()[1], p.size()[2])
i = image.cpu().data[0]   # image
g = color[0][:3]    #  ground truth


res = torch.cat( ( i, g, p), dim=2)
save_image( res, f"./Results/{uuid.uuid4().hex}.png")



t2 = time.time()
print( f"Done Successfully in: {t2-t1} seconds")


# img = image.cpu().data.numpy()[0].transpose( 1, 2, 0)
# pred = pred_class.float()/labels.size()[1]
# pred = pred.cpu().numpy()[0][0]



"""
plt.figure( figsize=(15, 30))
plt.subplot( 1, 3, 1)
plt.imshow( img)
plt.subplot( 1, 3, 2)
plt.imshow( pred)
plt.subplot( 1, 3, 3)
plt.imshow( color[0].numpy().transpose( 1, 2, 0))
plt.show()
"""
