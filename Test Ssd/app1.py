import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from tkinter import filedialog, Tk, Button, Canvas
from model.ssd import SimpleSSD
from PIL import ImageTk

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

class_names = ["gun", "knife"]
model = SimpleSSD()
model.load_state_dict(torch.load("ssd_weapon.pth", map_location="cpu"))
model.eval()

app = Tk()
app.title("Weapon Detection AI")
canvas = Canvas(app)
canvas.pack()
font = ImageFont.truetype("arial.ttf", 16)
tk_img = None
image_pil = None


def open_image():
    global image_pil, tk_img
    file_path = filedialog.askopenfilename()
    image_pil = Image.open(file_path).convert("RGB")
    tk_img = ImageTk.PhotoImage(image_pil.resize((400, 400)))
    canvas.config(width=400, height=400)
    canvas.create_image(0, 0, anchor="nw", image=tk_img)


def detect():
    global image_pil, tk_img
    if image_pil is None:
        return
    img_tensor = transform(image_pil).unsqueeze(0)
    with torch.no_grad():
        boxes, logits = model(img_tensor)
    boxes = boxes[0] * 256
    logits = F.softmax(logits[0], dim=1)
    image_draw = image_pil.resize((256, 256)).copy()
    draw = ImageDraw.Draw(image_draw)

    for i in range(3):
        conf, cls = torch.max(logits[i], dim=0)
        if conf < 0.5:
            continue
        x, y, w, h = boxes[i]
        left = x - w/2
        top = y - h/2
        right = x + w/2
        bottom = y + h/2
        draw.rectangle([left, top, right, bottom], outline="red", width=3)
        text = f"{class_names[cls]} ({conf:.2f})"
        try:
            text_size = font.getsize(text)
        except:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_size = (bbox[2]-bbox[0], bbox[3]-bbox[1])
        draw.rectangle([left, top - text_size[1], left + text_size[0], top], fill="red")
        draw.text((left, top - text_size[1]), text, fill="white", font=font)

    tk_img = ImageTk.PhotoImage(image_draw.resize((400, 400)))
    canvas.create_image(0, 0, anchor="nw", image=tk_img)

Button(app, text="Input", command=open_image, bg="#2ecc71", fg="white", width=15).pack(pady=10)
Button(app, text="Detect", command=detect, bg="#e74c3c", fg="white", width=15).pack(pady=5)
app.mainloop()