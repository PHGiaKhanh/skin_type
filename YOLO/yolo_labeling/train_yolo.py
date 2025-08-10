from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')
    model.train(
        data='./yolo_labeling/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='yolov8_face_parts',
        device='cuda'  # hoặc 'cpu'
    )

if __name__ == "__main__":
    main()
