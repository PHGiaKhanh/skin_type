# import os
# import cv2

# # Định nghĩa các vùng da mặt (normalized): (class_id, x_center, y_center, width, height)
# LABELS = [
#     (0, 0.45, 0.60, 0.25, 0.30),  # má (cheek)
#     (1, 0.50, 0.85, 0.30, 0.20),  # cằm (chin)
#     (2, 0.50, 0.20, 0.35, 0.30)   # trán (forehead)
# ]


# IMAGE_ROOT = "dataset_yolo/images"
# LABEL_ROOT = "dataset_yolo/labels"

# def write_yolo_label(img_path, label_path):
#     img = cv2.imread(img_path)
#     if img is None:
#         print(f"[!] Lỗi ảnh: {img_path}")
#         return
#     h, w = img.shape[:2]
#     os.makedirs(os.path.dirname(label_path), exist_ok=True)

#     with open(label_path, 'w') as f:
#         for cls_id, x_c, y_c, w_box, h_box in LABELS:
#             f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w_box:.6f} {h_box:.6f}\n")

# def process_all():
#     for subset in ['train', 'test']:
#         image_subset_dir = os.path.join(IMAGE_ROOT, subset)
#         label_subset_dir = os.path.join(LABEL_ROOT, subset)

#         for cls_name in os.listdir(image_subset_dir):  # dry, oily, normal
#             image_cls_dir = os.path.join(image_subset_dir, cls_name)
#             label_cls_dir = os.path.join(label_subset_dir, cls_name)

#             if not os.path.isdir(image_cls_dir):
#                 continue

#             for img_name in os.listdir(image_cls_dir):
#                 if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
#                     continue

#                 img_path = os.path.join(image_cls_dir, img_name)
#                 txt_name = os.path.splitext(img_name)[0] + ".txt"
#                 label_path = os.path.join(label_cls_dir, txt_name)

#                 write_yolo_label(img_path, label_path)
#                 print(f"[+] Tạo label: {subset}/{cls_name}/{img_name}")

# if __name__ == "__main__":
#     process_all()
import os
import cv2
import mediapipe as mp

# Cấu hình Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Định nghĩa class YOLO
CLASS_NAMES = {
    0: 'cheek',
    1: 'chin',
    2: 'forehead'
}

# Chỉ số landmark cho từng vùng
LANDMARK_INDEXES = {
    'cheek': [50, 101, 118, 202, 205, 425, 427, 430],  # vùng má ngoài, tránh môi
    'chin': [152, 200, 210, 199],
    'forehead': [10, 67, 69, 108, 109, 151]
}


def get_bounding_box_from_points(points, img_w, img_h, padding=0.05):
    x_coords = [int(p[0] * img_w) for p in points]
    y_coords = [int(p[1] * img_h) for p in points]

    x_min = max(min(x_coords) - int(padding * img_w), 0)
    y_min = max(min(y_coords) - int(padding * img_h), 0)
    x_max = min(max(x_coords) + int(padding * img_w), img_w)
    y_max = min(max(y_coords) + int(padding * img_h), img_h)

    # YOLO format: x_center, y_center, width, height (normalized)
    x_center = ((x_min + x_max) / 2) / img_w
    y_center = ((y_min + y_max) / 2) / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h

    return x_center, y_center, w, h

def process_image(img_path, label_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[!] Không đọc được ảnh: {img_path}")
        return

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        print(f"[!] Không phát hiện khuôn mặt: {img_path}")
        return

    face_landmarks = results.multi_face_landmarks[0].landmark
    os.makedirs(os.path.dirname(label_path), exist_ok=True)

    with open(label_path, "w") as f:
        for class_id, region_name in enumerate(['cheek', 'chin', 'forehead']):
            indices = LANDMARK_INDEXES[region_name]
            points = [(face_landmarks[i].x, face_landmarks[i].y) for i in indices]
            x_c, y_c, w_box, h_box = get_bounding_box_from_points(points, w, h)
            f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w_box:.6f} {h_box:.6f}\n")

    print(f"[+] Đã tạo label: {label_path}")

def process_all(root_dir="dataset_yolo/images", label_root="dataset_yolo/labels"):
    for subset in ["train", "valid"]:
        subset_path = os.path.join(root_dir, subset)
        for skin_type in os.listdir(subset_path):  # dry, oily, normal
            skin_path = os.path.join(subset_path, skin_type)
            if not os.path.isdir(skin_path):
                continue
            for file in os.listdir(skin_path):
                if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue
                img_path = os.path.join(skin_path, file)
                label_path = os.path.join(label_root, subset, skin_type, os.path.splitext(file)[0] + ".txt")
                process_image(img_path, label_path)

if __name__ == "__main__":
    process_all()

