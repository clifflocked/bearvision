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
epochs = 100

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

        blue_digits = [blue_score // 100, (blue_score % 100) // 10, blue_score % 10]
        red_digits = [red_score // 100, (blue_score % 100) // 100, blue_score % 10]

        blue_labels = torch.tensor(blue_digits, dtype=torch.long)
        red_labels = torch.tensor(red_digits, dtype=torch.long)

        return image, (blue_labels, red_labels)

class ScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 5 * 30, 256)

        self.fc_blue = nn.Linear(256, 30)
        self.fc_red = nn.Linear(256, 30)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        blue_out = self.fc_blue(x).view(-1, 3, 10)
        red_out = self.fc_red(x).view(-1, 3, 10)
        return blue_out, red_out

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ColorJitter(brightness=(0.75,1.25), contrast=(0.90, 1.10)),
        transforms.RandomAffine(2, scale=(0.95, 1.1)),
        transforms.ToTensor(),
    ])

    graph = open("./training/data.csv", "a")
    graph.write("index,loss_train,loss_val\n")

    dataset = ScoreDataset(transform=transform)

    val_dataset, train_dataset = random_split(dataset, [int(0.1 * len(dataset)), len(dataset) - int(0.1 * len(dataset))])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training data composed of {train_dataloader.__len__() * batch_size} images.")
    print(f"Validation data composed of {val_dataloader.__len__() * batch_size} images.")

    model = ScoreNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_dataloader:
            blue_labels, red_labels = labels

            optimizer.zero_grad()
            blue_out, red_out = model(images)

            loss_blue = sum(criterion(blue_out[:, i, :], blue_labels[:, i]) for i in range(3))
            loss_red = sum(criterion(red_out[:, i, :], red_labels[:, i]) for i in range(3))

            loss = loss_blue + loss_red

            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_dataloader:
                blue_labels, red_labels = labels
                blue_out, red_out = model(images)
                loss_blue = sum(criterion(blue_out[:, i, :], blue_labels[:, i]) for i in range(3))
                loss_red = sum(criterion(red_out[:, i, :], red_labels[:, i]) for i in range(3))
                val_loss = loss_blue + loss_red

        print(f"Epoch {epoch+1}, Training Loss: {loss.item():.6f}, Validation Loss: {val_loss.item():.6f}", end='\r')
        graph.write(f"0,{loss.item():.6f},{val_loss.item():.6f}\n")

    graph.close()
    print("\nDone!")
