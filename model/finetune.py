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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)
model.load_state_dict(torch.load("model/artifacts/best_model.pth", map_location=device))

# Selective unfreezing: only layer3, layer4, and fc become trainable
for name, param in model.named_parameters():
    if "layer3" in name or "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

model = model.to(device)

loss_fn = nn.CrossEntropyLoss()

# Different learning rates: gentle for the pretrained backbone layers, normal for fc
optimizer = torch.optim.Adam([
    {"params": model.layer3.parameters(), "lr": 1e-5},
    {"params": model.layer4.parameters(), "lr": 1e-5},
    {"params": model.fc.parameters(), "lr": 1e-4},
])

num_epochs = 5
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
        torch.save(model.state_dict(), "model/artifacts/best_model_finetuned.pth")
        print(f"  -> New best fine-tuned model saved (val_accuracy={val_accuracy:.4f})")

print(f"\nFine-tuning complete. Best validation accuracy: {best_val_accuracy:.4f}")