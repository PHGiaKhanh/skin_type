import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Global model
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

def advanced_face_crop_with_cleaning(image, padding=40):
    if image is None:
        return None

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ih, iw, _ = image.shape

    face_results = face_mesh.process(img_rgb)
    if not face_results.multi_face_landmarks:
        return None

    landmarks = face_results.multi_face_landmarks[0]

    # Viền khuôn mặt
    face_contour = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
                    365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172,
                    58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    face_points = np.array([[int(landmarks.landmark[i].x * iw),
                              int(landmarks.landmark[i].y * ih)]
                             for i in face_contour], dtype=np.int32)

    # Mask mặt
    mask = np.zeros((ih, iw), dtype=np.uint8)
    cv2.fillPoly(mask, [face_points], 255)

    # Xóa mắt và miệng
    regions = {
        "left_eye": [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
        "right_eye": [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382],
        "mouth": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314,
                  17, 84, 181, 91, 146, 61]
    }
    for region in regions.values():
        pts = np.array([[int(landmarks.landmark[i].x * iw),
                         int(landmarks.landmark[i].y * ih)]
                        for i in region], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 0)

    # Segmentation để loại nền
    seg = selfie_segmentation.process(img_rgb)
    seg_mask = (seg.segmentation_mask > 0.9).astype(np.uint8) * 255
    seg_mask = cv2.GaussianBlur(seg_mask, (7, 7), sigmaX=3)
    seg_mask = (seg_mask > 127).astype(np.uint8) * 255

    comb_mask = cv2.bitwise_and(mask, seg_mask)
    masked = cv2.bitwise_and(img_rgb, img_rgb, mask=comb_mask)

    gray_bg = np.full_like(img_rgb, fill_value=180)
    img_with_gray_bg = np.where(comb_mask[:, :, None] == 255, masked, gray_bg)

    # CROP + padding
    x0, y0 = face_points.min(axis=0)
    x1, y1 = face_points.max(axis=0)
    x0 = max(x0 - padding, 0)
    y0 = max(y0 - padding, 0)
    x1 = min(x1 + padding, iw)
    y1 = min(y1 + padding, ih)

    face_crop = img_with_gray_bg[y0:y1, x0:x1]
    if face_crop.size == 0:
        return None

    face_crop = cv2.resize(face_crop, (224, 224))
    return cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
