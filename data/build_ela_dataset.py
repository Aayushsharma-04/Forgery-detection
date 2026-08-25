import os
import pandas as pd
from tqdm import tqdm
from PIL import Image , ImageChops

def compute_ela(image_path,quality = 90):
   
    img  = Image.open(image_path).convert("RGB")
    img.save("temp.jpeg", "JPEG", quality = quality)
    temp_image = Image.open("temp.jpeg").convert("RGB")
    ela_image =ImageChops.difference(img,temp_image)
    extreme = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extreme])
    if max_diff ==0:
        max_diff =1
    scale =255.0/max_diff
    ela_image = ela_image.point(lambda p: p*scale)

    return ela_image

def main():

    manifest_path = "data/manifest.csv"
    output_dir = "data/processed"
    output_manifest = "data/processed_manifest.csv"
    os.makedirs(output_dir,exist_ok = True)

    df = pd.read_csv(manifest_path)
    processed_records =[]
    for index,row in tqdm(df.iterrows(),total = len(df)):
        file_path = row["filepath"]
        label = row["label"]

        filename = f"{label}_{os.path.basename(file_path)}"
        output_path = os.path.join(output_dir,filename)
        try:
            ela_image =compute_ela(file_path)
        
            ela_image.save(output_path, "JPEG")

            processed_records.append({"file_path" : output_path,"label" :label})
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    processed_df = pd.DataFrame(processed_records)
    processed_df.to_csv(output_manifest,index = False)

if __name__ =="__main__":
    main()
       
