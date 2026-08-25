
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torchvision  import transforms
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from data.transforms import train_transforms, val_test_transforms
def load_dataset():
    df = pd.read_csv("data/processed_manifest.csv")
    X = df["file_path"]
    y = df["label"]
    return X,y

def split_dataset():
    X,y = load_dataset()
    X_train, X_temp, y_train,y_temp = train_test_split(X,y,test_size = 0.3,random_state = 42,stratify = y)
    X_val, X_test ,y_val, y_test = train_test_split(X_temp,y_temp,test_size = 0.5,random_state = 42,stratify = y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test




class Elaimagedataset(Dataset):
    def __init__(self,dataframe,transform = None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self,idx):
        row = self.dataframe.iloc[idx]
        image_path = row["file_path"]
        label = row["label"]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image,label

X_train, X_val, X_test, y_train, y_val, y_test = split_dataset()


train_df = pd.DataFrame({"file_path": X_train, "label": y_train})
val_df = pd.DataFrame({"file_path": X_val, "label": y_val})
test_df = pd.DataFrame({"file_path": X_test, "label": y_test})


train_dataset = Elaimagedataset(train_df,transform = train_transforms)
val_dataset = Elaimagedataset(val_df,transform = val_test_transforms)
test_dataset = Elaimagedataset(test_df,transform = val_test_transforms)


if __name__ == "__main__":
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size:   {len(val_dataset)}")
    print(f"Test size:  {len(test_dataset)}")
 
    # Sanity check: pull one sample and confirm shape/type
    img, label = train_dataset[0]
    print(f"Sample image tensor shape: {img.shape}, label: {label}")




