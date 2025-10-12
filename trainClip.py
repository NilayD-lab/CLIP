# All CLIP is trying to do is map a text input and an image into the same latent space so that they are related. 
# So we are giving it some imaages and descriptons of images. If trained well, it should be able to take in an iamge
# and output a description of that image.  
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

import clip
import tqdm
import openpyxl
import os

from image_tileset import image_title_dataset
from util import *

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

xl_file = r"Training.xlsx"
wb = openpyxl.load_workbook(xl_file)

sheets = wb.sheetnames
ws = wb[sheets[0]]



# gathering entries from xl file
imgs = []
text = []
training_file = r"C:\Users\akani\Downloads\Training" # Training images folder is in the drive

not_found = 0
# loading up the image names and their descriptions
for i in range(1, ws.max_row + 1): 
    path = str(ws.cell(i, 1).value)
    img_name = path.split("/")[-1]
    value = str(ws.cell(i, 2).value)
    if (os.path.exists(training_file + "/" + img_name)):
        imgs.append(training_file + "/" + img_name)
        text.append(value)
        extra_descriptions = combine_descriptions(ws, i) # here is where im combing the extra descriptions i annotated. Removing may lead to better
        # results but that is still shaky
        if (extra_descriptions is not None):
            text.append(extra_descriptions)
            imgs.append(training_file + "/" + img_name)
            # print(extra_descriptions)
        
print(f"amount of images: {len(imgs)}")
print(f"amount of images: {len(text)}")
print(f"not found: {not_found}")
print(text[:5])

exit(1)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

dataset = image_title_dataset(imgs, text, preprocess) # Here we create an object that will take care of loading an image
# the obejct is mainly so we can pass something into the DataLoader class
train_dataloader = DataLoader(dataset, batch_size=20, shuffle=True) # try reducing batch size if you run out of memory

num_epochs = 200
for epoch in range(num_epochs):
    pbar = tqdm.tqdm(train_dataloader, total=len(train_dataloader))
    for batch in pbar:
        optimizer.zero_grad()

        images,texts = batch

        images= images.to(device)
        texts = texts.to(device)

        # Forward pass
        logits_per_image, logits_per_text = model(images, texts)
        # Compute loss
        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2

        # Backward pass
        total_loss.backward()
        if device == "cpu":
            optimizer.step()
        else :
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")



torch.save(model.state_dict(), f"C:/Users/akani/Downloads/clip_model_finetuned_{num_epochs}.pth")
