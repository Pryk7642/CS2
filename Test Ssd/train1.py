import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from model.ssd import SimpleSSD

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

class_to_idx = {"gun": 0, "knife": 1, "bom": 2}

class WeaponDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = transform(img)
        box = torch.tensor([
            float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"])
        ], dtype=torch.float32)
        label = class_to_idx[row["class"]]
        return img, box, label

# โหลด CSV
df = pd.read_csv("dataset/labels.csv")
dataset = WeaponDataset(df)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleSSD(num_classes=3).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_box = nn.MSELoss()
loss_cls = nn.CrossEntropyLoss()

for epoch in range(100):
    total_loss = 0
    for imgs, boxes, labels in loader:
        imgs = imgs.to(DEVICE)
        boxes = boxes.unsqueeze(1).expand(-1, 3, -1).to(DEVICE)
        labels = labels.to(DEVICE)

        pred_boxes, pred_logits = model(imgs)
        loss1 = loss_box(pred_boxes, boxes)
        loss2 = 0
        for i in range(pred_logits.shape[1]):
            loss2 += loss_cls(pred_logits[:, i], labels)

        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/100 - Loss: {total_loss:.4f}")


torch.save(model.state_dict(), "ssd_weapon.pth")
