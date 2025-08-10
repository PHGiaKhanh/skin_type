
# import os
# import cv2
# import numpy as np
# from tqdm import tqdm
# from pathlib import Path
# import mediapipe as mp

# mp_face_mesh = mp.solutions.face_mesh

# REGION_LANDMARKS = {
#     "cheeks": [
#         # Má trái và phải (vùng U-zone)
#         50, 101, 118, 119, 120, 121, 202,      # Má trái
#         280, 347, 346, 330, 329, 348, 450      # Má phải
#     ],
#     "forehead": [
#         #chuẩn đẹp
#         10, 338, 297, 332, 284, 251, 389, 356,
#         70, 63, 105, 66, 107, 9
#     ],
#     "chin": [
#         # Vùng cằm + dưới miệng
#         152, 377, 400, 378, 379, 365,
#         288, 435, 427, 424, 367, 364
#     ]
# }



# def extract_region(image, region="cheeks", padding=10):
#     with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1) as face_mesh:
#         if not isinstance(image, np.ndarray):
#             image = np.array(image)

#         img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(img_rgb)

#         if not results.multi_face_landmarks:
#             return None

#         landmarks = results.multi_face_landmarks[0]
#         h, w = image.shape[:2]

#         indices = REGION_LANDMARKS.get(region)
#         if not indices:
#             raise ValueError(f"Unknown region: {region}")

#         points = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in indices]
#         points_np = np.array(points)

#         x, y, w_box, h_box = cv2.boundingRect(points_np)
#         if w_box < 30 or h_box < 30:
#             return None

#         x = max(x - padding, 0)
#         y = max(y - padding, 0)
#         x2 = min(x + w_box + padding, image.shape[1])
#         y2 = min(y + h_box + padding, image.shape[0])

#         return image[y:y2, x:x2]


# # Ghép 3 vùng theo chiều dọc
# def process_image(image_path):
#     img = cv2.imread(str(image_path))
#     if img is None:
#         return None

#     regions = []
#     for region in ["forehead", "cheeks", "chin"]:
#         crop = extract_region(img, region)
#         if crop is None:
#             return None
#         regions.append(crop)

#     # Resize các vùng theo cùng chiều rộng
#     target_width = max(r.shape[1] for r in regions)
#     resized = [
#         cv2.resize(r, (target_width, int(r.shape[0] * target_width / r.shape[1])))
#         for r in regions
#     ]

#     combined = cv2.vconcat(resized)
#     combined = cv2.resize(combined, (224, 224))
#     return combined

# # Xử lý hàng loạt ảnh
# def batch_process(input_root, output_root):
#     input_root = Path(input_root)
#     output_root = Path(output_root)
#     output_root.mkdir(parents=True, exist_ok=True)

#     for split in ['train', 'valid', 'test']:
#         for label in ['dry', 'normal', 'oily']:
#             in_dir = input_root / split / label
#             out_dir = output_root / split / label
#             out_dir.mkdir(parents=True, exist_ok=True)

#             image_paths = list(in_dir.glob("*.jpg")) + list(in_dir.glob("*.png")) + list(in_dir.glob("*.jpeg"))

#             for img_path in tqdm(image_paths, desc=f"[{split}/{label}]"):
#                 try:
#                     combined_img = process_image(img_path)
#                     if combined_img is None:
#                         continue
#                     out_path = out_dir / img_path.name
#                     cv2.imwrite(str(out_path), combined_img)
#                 except Exception as e:
#                     print(f"❌ Error {img_path.name}: {e}")

# # Chạy
# if __name__ == "__main__":
#     batch_process("eyes_mouth", "combined_eyes_224_verti")
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp

# ---------- LANDMARK CỦA TỪNG VÙNG ----------
REGION_LANDMARKS = {
    "forehead": [
        10, 338, 297, 332, 284, 251, 389, 356,
        70, 63, 105, 66, 107, 9
    ],
    "cheeks": [
        234, 93, 132, 58,         # má trái
        361, 323, 356, 288,       # má phải
        205, 187                  # trung tâm (loại bỏ các điểm gần miệng)
    ],

    "chin": [
        152, 200, 210, 211, 429  # Tập trung vùng dưới miệng, chính giữa
    ]
        
}

# ---------- CẮT MỖI VÙNG TỪ LANDMARK ----------
def extract_region(image, region="cheeks", padding_ratio=0.1):
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return None

    mp_face_mesh = mp.solutions.face_mesh
    mp_detection = mp.solutions.face_detection

    with mp_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6) as detector:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        det_result = detector.process(rgb)
        if not det_result.detections:
            print(f"❌ No face detected in image for region {region}")
            return None

        bbox = det_result.detections[0].location_data.relative_bounding_box
        h, w = image.shape[:2]
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)
        face_crop = image[y1:y2, x1:x2]

    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1) as face_mesh:
        rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_face)

        if not results.multi_face_landmarks:
            print(f"❌ No landmarks found for region {region}")
            return None

        landmarks = results.multi_face_landmarks[0]
        h_f, w_f = face_crop.shape[:2]
        indices = REGION_LANDMARKS.get(region)
        if not indices:
            raise ValueError(f"Unknown region: {region}")

        points = [(int(landmarks.landmark[i].x * w_f), int(landmarks.landmark[i].y * h_f)) for i in indices]
        points_np = np.array(points)
        x, y, w_box, h_box = cv2.boundingRect(points_np)

        # padding
        x_pad = int(w_box * padding_ratio)
        y_pad_top = int(h_box * padding_ratio)
        y_pad_bottom = int(h_box * padding_ratio)

        # Điều chỉnh riêng từng vùng
        if region == "cheeks":
            y_pad_top = int(h_box * 0.05)
            y_pad_bottom = int(h_box * 0.02)  # rất nhỏ, tránh miệng

        elif region == "chin":
            y_pad_top = int(h_box * 0.005)
            y_pad_bottom = int(h_box * 0.15)
            shift = int(h_box * 0.1)
            y = min(y + shift, face_crop.shape[0] - 1)


        elif region == "forehead":
            y_pad_bottom = int(h_box * 0.15)

        x = max(x - x_pad, 0)
        y = max(y - y_pad_top, 0)
        x2 = min(x + w_box + 2 * x_pad, face_crop.shape[1])
        y2 = min(y + h_box + y_pad_top + y_pad_bottom, face_crop.shape[0])

        return face_crop[y:y2, x:x2]

# ---------- XỬ LÝ 1 ẢNH ----------
def process_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Failed to load image {image_path}")
        return None

    regions = []
    for region in ["forehead", "cheeks", "chin"]:
        crop = extract_region(img, region)
        if crop is not None:
            regions.append(crop)
        else:
            print(f"⚠️ Failed to extract {region} from {image_path}")

    if not regions:
        print(f"❌ No valid regions extracted from {image_path}")
        return None

    # Resize theo chiều rộng giống nhau để ghép dọc đẹp
    target_width = max(r.shape[1] for r in regions)
    resized = [
        cv2.resize(r, (target_width, int(r.shape[0] * target_width / r.shape[1])))
        for r in regions
    ]
    combined = cv2.vconcat(resized)

    # Resize về 224x224
    h, w = combined.shape[:2]
    if h > w:
        new_h, new_w = 224, int(w * 224 / h)
    else:
        new_w, new_h = 224, int(h * 224 / w)

    combined = cv2.resize(combined, (new_w, new_h))
    padded = np.zeros((224, 224, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = combined
    return padded

# ---------- XỬ LÝ TOÀN BỘ DỮ LIỆU ----------
def batch_process(input_root, output_root):
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'valid', 'test']:
        for label in ['dry', 'normal', 'oily']:
            in_dir = input_root / split / label
            out_dir = output_root / split / label
            out_dir.mkdir(parents=True, exist_ok=True)

            if not in_dir.exists():
                print(f"⛔ Folder {in_dir} not found, skipping...")
                continue

            image_paths = list(in_dir.glob("*.jpg")) + list(in_dir.glob("*.png")) + list(in_dir.glob("*.jpeg"))
            for img_path in tqdm(image_paths, desc=f"[{split}/{label}]"):
                try:
                    combined_img = process_image(img_path)
                    if combined_img is None:
                        continue
                    out_path = out_dir / img_path.with_suffix(".jpg").name
                    cv2.imwrite(str(out_path), combined_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                except Exception as e:
                    print(f"❌ Error with {img_path.name}: {e}")

# ---------- CHẠY ----------
if __name__ == "__main__":
    batch_process("eyes_mouth", "crop_media_new2")
