from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torchvision import models as models
from data.dataset import test_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

BATCH_SIZE = 32
test_loader = DataLoader(test_dataset,batch_size =BATCH_SIZE,shuffle = False,num_workers =0)
model = models.resnet18(weights = None)
num_features= model.fc.in_features
model.fc = nn.Linear(num_features,2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("model/artifacts/best_model_finetuned.pth",map_location = device))

model = model.to(device)
model.eval()
all_preds =[]
all_labels =[]

with torch.no_grad():
    for X_batch,y_batch in tqdm(test_loader,desc = "Collecting predictions"):
        X_batch,y_batch = X_batch.to(device),y_batch.to(device)
        pred = model(X_batch)
        predicted = torch.argmax(pred,dim =1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())



cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)
print("(rows = actual class, columns = predicted class)")
print("(row 0 = authentic, row 1 = tampered)")


report = classification_report(all_labels, all_preds, target_names=["authentic", "tampered"])
print("\nClassification Report:")
print(report)


plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["authentic", "tampered"],
    yticklabels=["authentic", "tampered"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Fine-tuned Model")
plt.tight_layout()
plt.savefig("model/artifacts/confusion_matrix.png")
print("\nConfusion matrix image saved to model/artifacts/confusion_matrix.png")
