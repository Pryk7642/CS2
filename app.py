import torch
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
from torchvision import transforms
from model.model import WeaponDetector

# คลาส
classes = ["gun", "knife", "bom"]

# โหลดโมเดล
model = WeaponDetector(num_classes=len(classes))
model.load_state_dict(torch.load("weapon_detector.pth", map_location="cpu"))
model.eval()

# การแปลงภาพ
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# สร้างหน้าต่างแอพ
root = tk.Tk()
root.title("Weapon Detector")
root.geometry("500x500")
root.configure(bg="#2c3e50")

original_img = None
display_img = None
tk_img = None

frame_btn = tk.Frame(root, bg="#2c3e50")
frame_btn.pack(pady=15)

canvas = tk.Canvas(root, width=256, height=256, bg="#34495e", highlightthickness=0)
canvas.pack(pady=10)

# โหลดตัวอหนังสือสำหรับแสดง
try:
    font = ImageFont.truetype("arial.ttf", 15)
except:
    font = ImageFont.load_default()

# ปุ่มเลือกภาพ
def load_image():
    global original_img, display_img, tk_img
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return
    original_img = Image.open(file_path).convert("RGB")
    display_img = original_img.resize((256, 256))
    tk_img = ImageTk.PhotoImage(display_img)
    canvas.config(width=tk_img.width(), height=tk_img.height())
    canvas.delete("all")
    canvas.create_image(0, 0, anchor="nw", image=tk_img)

# ปุ่มตรวจจับ
def detect():
    global display_img, tk_img
    if display_img is None:
        print("ยังไม่มีภาพโหลดเข้ามา กรุณากด Input ก่อน")
        return

    img_tensor = transform(display_img).unsqueeze(0)
    
    with torch.no_grad():
        pred_bbox, pred_class_logits = model(img_tensor)

        pred_bbox = pred_bbox[0]
        pred_class = F.softmax(pred_class_logits, dim=1)[0]
        class_idx = torch.argmax(pred_class).item()
        class_name = classes[class_idx]

        # คูณกลับเป็นขนาดภาพ
        x, y, w, h = pred_bbox
        x *= 256
        y *= 256
        w *= 256
        h *= 256

    # วาดกรอบและชื่อ
    img_with_box = display_img.copy()
    draw = ImageDraw.Draw(img_with_box)

    try:
        text_size = font.getsize(class_name)
    except AttributeError:
        bbox = draw.textbbox((0, 0), class_name, font=font)
        text_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])

    draw.rectangle([(x - w/2, y - h/2), (x + w/2, y + h/2)], outline="#e74c3c", width=3)
    text_bg_rect = [x - w/2, y - h/2 - text_size[1] - 5, x - w/2 + text_size[0] + 6, y - h/2]
    draw.rectangle(text_bg_rect, fill="#e74c3c")
    draw.text((x - w/2 + 3, y - h/2 - text_size[1] - 3), class_name, fill="white", font=font)

    tk_img = ImageTk.PhotoImage(img_with_box)
    canvas.config(width=tk_img.width(), height=tk_img.height())
    canvas.delete("all")
    canvas.create_image(0, 0, anchor="nw", image=tk_img)

# ปุ่ม
btn_load = tk.Button(frame_btn, text="Input Image", command=load_image,
                     bg="#2980b9", fg="white", font=("Arial", 12, "bold"),
                     width=12, height=2, relief="flat", activebackground="#3498db")
btn_load.pack(side="left", padx=10)

btn_detect = tk.Button(frame_btn, text="Detect", command=detect,
                       bg="#27ae60", fg="white", font=("Arial", 12, "bold"),
                       width=12, height=2, relief="flat", activebackground="#2ecc71")
btn_detect.pack(side="left", padx=10)

# เริ่ม
root.mainloop()
