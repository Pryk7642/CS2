import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image
from torchvision import transforms
from model.model import WeaponDetector

# กำหนดค่าเทรน
EPOCHS = 200
BATCH = 8
IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# การแปลงภาพก่อนป้อนเข้าโมเดล
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(), 
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])

# โหลด label dataset จาก CSV
df = pd.read_csv("dataset/labels.csv")

# กำหนด class
classes = ["gun", "knife", "bom"]
class_to_idx = {c: i for i, c in enumerate(classes)}

# Dataset Class สำหรับโหลดภาพและ label
class WeaponDataset(torch.utils.data.Dataset):
    def __init__(self, df, class_to_idx):
        self.df = df
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = transform(img)

        # โหลดข้อมูลตำแหน่งกรอบ
        box = torch.tensor([
            float(str(row["x"]).strip()),
            float(str(row["y"]).strip()),
            float(str(row["w"]).strip()),
            float(str(row["h"]).strip())
        ], dtype=torch.float32)

        label = self.class_to_idx[row["class"].strip()]
        return img, box, label

# สร้าง Dataset และ DataLoader
dataset = WeaponDataset(df, class_to_idx)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=True)

# สร้างโมเดล
model = WeaponDetector(num_classes=len(classes)).to(DEVICE)

# loss function
criterion_bbox = nn.MSELoss()  # สำหรับตำแหน่งกรอบ
criterion_class = nn.CrossEntropyLoss()  # สำหรับคลาส

# optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# เริ่มเทรน
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, boxes, labels in loader:
        imgs, boxes, labels = imgs.to(DEVICE), boxes.to(DEVICE), labels.to(DEVICE)

        # ทำนายจากภาพ
        preds_bbox, preds_class = model(imgs)

        # คำนวณ loss
        loss_bbox = criterion_bbox(preds_bbox, boxes)
        loss_class = criterion_class(preds_class, labels)
        loss = loss_bbox + loss_class

        # อัปเดตพารามิเตอร์
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

# บันทึกโมเดล
torch.save(model.state_dict(), "weapon_detector.pth")
print("Saved model Successful!!")
