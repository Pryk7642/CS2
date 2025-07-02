import torch
import torch.nn as nn
import torch.nn.functional as F

class WeaponDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(WeaponDetector, self).__init__()
        
        # convolutional layer แรก: รับภาพ RGB → 16 channel
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        
        # convolutional layer ที่สอง: 16 channel → 32 channel
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        
        # max pooling ลดขนาด feature map ลงครึ่งหนึ่ง
        self.pool = nn.MaxPool2d(2, 2)
        
        # flatten แล้วเข้าสู่ fully connected layer (หลัง pool 2 รอบ → 64x64)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)
        
        # output สำหรับพยากรณ์ตำแหน่งกรอบ (bounding box) [x, y, w, h]
        self.fc_bbox = nn.Linear(128, 4)
        
        # output สำหรับการจำแนกคลาสอาวุธ (class logits)
        self.fc_class = nn.Linear(128, num_classes)

    def forward(self, x):
        # ผ่าน conv + relu + pool
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # flatten เพื่อเข้าสู่ fully connected layer
        x = x.view(-1, 32 * 64 * 64)
        x = F.relu(self.fc1(x))
        
        # พยากรณ์ bbox และ class
        bbox = torch.sigmoid(self.fc_bbox(x))  # ใช้ sigmoid ให้อยู่ในช่วง 0-1
        class_logits = self.fc_class(x)
        
        return bbox, class_logits
