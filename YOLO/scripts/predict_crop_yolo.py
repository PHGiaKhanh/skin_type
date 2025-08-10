
from ultralytics import YOLO
import cv2
import os

# Load YOLOv8 model đã huấn luyện
model = YOLO("runs/detect/yolov8_face_parts/weights/best.pt")

# Class mapping (giống data.yaml)
CLASS_NAMES = {
    0: 'cheeks',
    1: 'chin',
    2: 'forehead'
}

def crop_and_save(image_path, split, label_name, save_root="crop_yolo"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[!] Không đọc được ảnh: {image_path}")
        return

    results = model(img)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        region = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"[!] Crop {region} bị rỗng.")
            continue

        # Đường dẫn: crop_yolo/train/cheeks/normal/img_xxx.jpg
        filename = os.path.basename(image_path)
        filename_out = os.path.splitext(filename)[0] + f".jpg"

        save_dir = os.path.join(save_root, split, region, label_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename_out)

        cv2.imwrite(save_path, crop)
        print(f"[✓] Saved: {save_path}")
if __name__ == "__main__":
    input_root = "dataset_yolo/images"    # data/raw_images/train/dry/image.jpg
    output_root = "crop_yolo"         # output crop folder

    splits = ['train', 'valid']

    for split in splits:
        split_dir = os.path.join(input_root, split)
        if not os.path.isdir(split_dir):
            continue

        for label in os.listdir(split_dir):  # dry, oily, normal
            label_dir = os.path.join(split_dir, label)
            if not os.path.isdir(label_dir):
                continue

            for file in os.listdir(label_dir):
                if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img_path = os.path.join(label_dir, file)
                crop_and_save(img_path, split=split, label_name=label, save_root=output_root)
