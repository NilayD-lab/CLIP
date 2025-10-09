import clip
from PIL import Image


class image_title_dataset():
    def __init__(self, list_image_path, list_txt, preprocess):
        # Initialize image paths and corresponding texts
        self.image_path = list_image_path
        # Tokenize text using CLIP's tokenizer
        self.title = clip.tokenize(list_txt)
        self.preprocess = preprocess
    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        # Preprocess image using CLIP's preprocessing function
        image = self.preprocess(Image.open(self.image_path[idx]))
        title = self.title[idx]
        return image, title

