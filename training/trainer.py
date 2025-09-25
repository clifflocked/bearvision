import json
import glob
import os
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

batch_size = 16
epochs = 500

class ScoreDataset(Dataset):
    def __init__(self, img_dir="samples/scores/images", json_dir="samples/scores/data", transform=None):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.json_paths = sorted(glob.glob(os.path.join(json_dir, "*.json")))
        self.transform = transform

        assert len(self.img_paths) == len(self.json_paths), "Images and data don't match"

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        json_path = self.json_paths[idx]
        with open(json_path, "r") as f:
            data = json.load(f)

        blue_score = int(data["bluescore"])
        red_score = int(data["redscore"])

        labels = torch.tensor([blue_score, red_score], dtype=torch.long)

        return image, labels
    
class ScoreNet(nn.Module):
    def __init__(self, num_classes=401):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 5 * 30, 256)

        self.fc_blue = nn.Linear(256, num_classes)
        self.fc_red = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        blue_out = self.fc_blue(x)
        red_out = self.fc_red(x)
        return blue_out, red_out

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=(0.75,1.25), contrast=(0.90, 1.10)),
        transforms.RandomAffine(2, scale=(0.95, 1.1)),
        transforms.ToTensor(),
    ])

    dataset = ScoreDataset(transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Training data composed of {dataloader.__len__() * batch_size} images.")

    model = ScoreNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for images, labels in dataloader:
            blue_labels = labels[:, 0]
            red_labels = labels[:, 1]

            optimizer.zero_grad()
            blue_out, red_out = model(images)

            loss_blue = criterion(blue_out, blue_labels)
            loss_red = criterion(red_out, red_labels)
            loss = loss_blue + loss_red

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}", end='\r')
    
    print("\nDone!")