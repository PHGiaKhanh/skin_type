import cv2
import os
from glob import glob
from new_crop import advanced_face_crop_with_cleaning  # import hàm crop nếu bạn tách riêng file

# === Cấu hình đường dẫn ===
input_root = "data"              
output_root = "eyes_mouth"        

splits = ["train", "valid", "test"]
labels = ["dry", "oily", "normal"]
exts = [".jpg", ".jpeg", ".png"]


def process_dataset():
    for split in splits:
        for label in labels:
            input_dir = os.path.join(input_root, split, label)
            output_dir = os.path.join(output_root, split, label)
            os.makedirs(output_dir, exist_ok=True)

            if not os.path.exists(input_dir):
                print(f"⚠️ Bỏ qua {input_dir} (không tồn tại)")
                continue

            image_paths = glob(os.path.join(input_dir, "*"))
            for img_path in image_paths:
                fname = os.path.basename(img_path)
                name, ext = os.path.splitext(fname)
                if ext.lower() not in exts:
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    print(f"❌ Không đọc được: {img_path}")
                    continue

                face = advanced_face_crop_with_cleaning(img)
                if face is None:
                    print(f"⚠️ Không phát hiện mặt: {img_path}")
                    continue

                save_path = os.path.join(output_dir, f"{name}.jpg")
                cv2.imwrite(save_path, face)
                print(f"✅ Saved: {save_path}")


if __name__ == "__main__":
    process_dataset()
