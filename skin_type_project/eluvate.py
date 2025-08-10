import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from models.efficientnet_model import get_model
from utils.transforms import get_transforms  # Dùng đúng transform train/val

# ---------- Cấu hình ----------
model_path = "best_model2.pth"
#image_path = "data/test/normal/normal_0a973116bdfa8386999e_jpg.rf.0ea979e1616c011bd977dfc4defd88c9.jpg"  
#image_path = "crop_yolo_new2/valid/normal/normal_0a973116bdfa8386999e_jpg.rf.7c680f8d491a8acdc7623063eff9102f.jpg"
image_path = "crop_yolo_new2/valid/dry/dry_1f55332514604c83ba14_jpg.rf.026140eee130dcb4368bf7e895fb9500.jpg"
#image_path = "crop_yolo_new2/valid/oily/oily_2cdb581b88b7e8fd4441_jpg.rf.14c68bcfe99df77bb688a6593e40ab61.jpg"

class_names = ["dry", "normal", "oily"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Load mô hình ----------
model = get_model(num_classes=3).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ---------- Load & tiền xử lý ảnh ----------
transform = get_transforms(train=False)
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)  # shape: (1, C, H, W)

# ---------- Dự đoán + độ tin cậy ----------
with torch.no_grad():
    output = model(input_tensor)                 # logits (trước softmax)
    probs = F.softmax(output, dim=1)[0]          # xác suất sau softmax
    pred_idx = torch.argmax(probs).item()
    predicted_class = class_names[pred_idx]
    confidence = probs[pred_idx].item()

# ---------- Kết quả ----------
print(f" Ảnh: {image_path}")
print(f" Dự đoán: {predicted_class} (Độ tin cậy: {confidence * 100:.2f}%)")
