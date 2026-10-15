
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import cv2
from torchvision import transforms

def preprocess_face(image_bgr):
    # Khử nhiễu giữ biên bằng Bilateral Filter
    image_bgr = cv2.bilateralFilter(image_bgr, d=9, sigmaColor=75, sigmaSpace=75)

    # Tăng tương phản vùng sáng (đặc biệt quan trọng cho da mặt)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(16, 16))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    image_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Làm sắc nét ảnh (Unsharp Masking)
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(image_bgr, 1.5, blurred, -0.5, 0)

    return sharpened


class SkinType(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.transform = transform
        self.categories = ["dry", "normal", "oily"]
        self.image_paths, self.labels = [], []

        for label, category in enumerate(self.categories):
            category_path = os.path.join(root, split, category)
            if not os.path.exists(category_path):
                raise FileNotFoundError(f"❌ Folder không tồn tại: {category_path}")
            for fname in os.listdir(category_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(category_path, fname)
                    if os.path.isfile(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx, attempts=0):
        if attempts > 5:
            raise RuntimeError(f"❌ Quá nhiều lỗi liên tiếp khi load ảnh tại index {idx}")

        try:
            # Load ảnh RGB bằng PIL
            image = Image.open(self.image_paths[idx]).convert("RGB")
            image_np = np.array(image)

            # Chuyển sang BGR để dùng với OpenCV
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Tiền xử lý ảnh mặt
            image_bgr = preprocess_face(image_bgr)

            # Chuyển lại về RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_rgb)

            # Áp dụng transform torchvision (nếu có)
            if self.transform:
                image = self.transform(image)

            return image, self.labels[idx]

        except Exception as e:
            print(f"⚠️ Lỗi ảnh tại {self.image_paths[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self), attempts + 1)
