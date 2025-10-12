# I took 20 of my annotations and seperated them in a differente excel file
# These will serve has the testing set for the model
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

import clip
import tqdm
import openpyxl
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


model, preprocess = clip.load("ViT-B/32", device=device)
model.load_state_dict(torch.load(r"C:\Users\akani\Downloads\clip_model_finetuned_200.pth", map_location=device))
model.eval()


new_path = r"C:\Users\akani\Downloads\Nilay\final"
imgs = []
classes  = []           
xl_file = r"testCLIP.xlsx"
wb = openpyxl.load_workbook(xl_file)
ws = wb.active    

# gathering entries from xl file
for i in range(1, ws.max_row + 1): 
    path = str(ws.cell(i, 1).value)
    img_name = path.split("/")[-1]
    value = str(ws.cell(i, 2).value)
    if (os.path.exists(new_path + "/" + img_name)):
        imgs.append(new_path + "/" + img_name)
        classes.append(value.lower())

# some garbage prompts to add
classes.append("Dr.Limestone cracked their back on the brick wall")
classes.append("Stepped into the crack under the balcony")
for i in range(len(imgs)):
    # preprcess featrures so they are in the same format as CLIP training
    img = Image.open(imgs[i])
    image_input = preprocess(img).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(c) for c in classes]).to(device) # create the prompts to test

    # encode those features by passsing through the model
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
    # normalize features
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1) # calculate similarity
    # Pick the top k most similar labels for the image
    values, indices = similarity[0].topk(len(classes))

    # Print the result
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
    print(f"actual description: {classes[i]}")
