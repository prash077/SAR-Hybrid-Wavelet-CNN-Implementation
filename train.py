import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import glob
import os
import matplotlib.pyplot as plt

from config import TRAIN_DIR, BATCH_SIZE, PATCH_SIZE, EPOCHS, LEARNING_RATE, PATCHES_PER_FILE, MODEL_SAVE_PATH
from dataset import SARDatasetOptimized
from model import SimpleSAR_CNN

def train():
    train_files = glob.glob(os.path.join(TRAIN_DIR, "*.zip"))
    if not train_files:
        print(f" Error: No training files found in {TRAIN_DIR}")
        return

    dataset = SARDatasetOptimized(train_files, patch_size=PATCH_SIZE, patches_per_file=PATCHES_PER_FILE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    model = SimpleSAR_CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    loss_history = []
    
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.savefig("training_loss.png")
    print("Graph saved as training_loss.png")

if __name__ == "__main__":
    train()