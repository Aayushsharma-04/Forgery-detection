import torch
from torch.utils.data import DataLoader
from torchvision import models as models
from data.dataset import train_dataset, val_dataset, test_dataset
import torch.nn as nn
from tqdm import tqdm

BATCH_SIZE = 32
NUM_WORKERS = 0
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

model = models.resnet18(weights="IMAGENET1K_V1")

for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

for param in model.fc.parameters():
    param.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

num_epochs = 10
best_val_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1} training"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = 0
        correct = 0
        total = 0
        for X_batch, y_batch in tqdm(val_loader, desc=f"Epoch {epoch+1} validation"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            val_loss += loss.item()
            predicted = torch.argmax(pred, dim=1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        print(f"Epoch {epoch+1}: val_loss = {avg_val_loss:.4f}, val_accuracy = {val_accuracy:.4f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "model/artifacts/best_model.pth")
        print(f"  -> New best model saved (val_accuracy={val_accuracy:.4f})")

print(f"\nTraining complete. Best validation accuracy: {best_val_accuracy:.4f}")

