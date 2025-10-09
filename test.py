
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
model.load_state_dict(torch.load(r"C:\Users\akani\Downloads\clip_model_finetuned_300.pth", map_location=device))
model.eval()


new_path = r"C:\Users\akani\Downloads\Nilay\final"


image = Image.open(new_path + "\\" + "1036877_9C-I1_DetailedPhoto.pdf_1_1.jpeg")
image_input = preprocess(image).unsqueeze(0).to(device)
classes = ["crack on white brick parapet", 
           "crack on concrete coping"
           "cracked stucco wall",
           "cracked parapet wall", 
           "Cracked mortar on the limestone near the window", 
           "Corner masonry cracking", 
           "Cracked Sill on South Elevation", 
           "Muiltiple cracks on brick masonry", 
           "Hairline crack in brickwork on 17th floor",
       
           "Dr.Limestone cracked their back on the brick wall",
           "Stepped into the crack under the balcony"] # provide text classes for it to guess on
text_inputs = torch.cat([clip.tokenize(f"a photo of {c}") for c in classes]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(len(classes))

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")

# torch.save(model.state_dict(), r"C:\Users\akani\Downloads\clip_model_finetuned.pth")
