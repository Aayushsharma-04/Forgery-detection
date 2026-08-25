
from PIL import Image ,ImageChops
import numpy as np


def compute_ela(image_path,quality = 90):

    if isinstance(image_path, str):
        img = Image.open(image_path).convert("RGB")
    else:
        img = image_path.convert("RGB")

   
    
    temp = img.save("temp.jpeg", "JPEG", quality = quality)
    temp_image = Image.open("temp.jpeg").convert("RGB")
    ela_image =ImageChops.difference(img,temp_image)
    extreme = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extreme])
    if max_diff ==0:
        max_diff =1;
    scale =255.0/max_diff
    ela_image = ela_image.point(lambda p: p*scale)

    return ela_image



