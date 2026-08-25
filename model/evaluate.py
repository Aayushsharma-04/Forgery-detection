import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import models as models
from data.dataset import test_dataset
from torch import nn as nn

BATCH_SIZE = 32
test_loader = DataLoader(test_dataset,batch_size = BATCH_SIZE,shuffle = False,num_workers = 0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights = None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features,2)
model.load_state_dict(torch.load("model/artifacts/best_model.pth",map_location = device))
model = model.to(device)
model.eval()
loss_fn = nn.CrossEntropyLoss()

with torch.no_grad():
    test_loss =0
    correct =0
    total =0
    for X_batch,y_batch in tqdm(test_loader,desc = "Testing"):
        X_batch,y_batch = X_batch.to(device),y_batch.to(device)
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        test_loss += loss.item()
        predicted = torch.argmax(pred, dim=1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = correct / total
    print(f" test_loss = {avg_test_loss:.4f}, test_accuracy = {test_accuracy:.4f}")



 