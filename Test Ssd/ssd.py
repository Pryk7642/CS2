import torch
import torch.nn as nn

class SimpleSSD(nn.Module):
    def __init__(self, num_boxes=3, num_classes=3):
        super().__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 64 * 64, 512),
            nn.ReLU(),
        )
        self.bbox_head = nn.Linear(512, num_boxes * 4)
        self.class_head = nn.Linear(512, num_boxes * num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        bbox = self.bbox_head(x).view(-1, self.num_boxes, 4)
        class_logits = self.class_head(x).view(-1, self.num_boxes, self.num_classes)
        return bbox, class_logits
