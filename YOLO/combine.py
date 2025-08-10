import os
from PIL import Image

base_dir = 'crop_yolo_new2'
sets = ['train','valid']
regions = ['forehead', 'cheeks', 'chin']

for set_name in sets:
    save_dir = os.path.join(base_dir, f"{set_name}_combined")

    region_dir = os.path.join(base_dir, set_name, 'forehead')
    for label in os.listdir(region_dir):
        label_path = os.path.join(region_dir, label)
        if not os.path.isdir(label_path):
            continue

        # Tạo thư mục lớp bên trong thư mục combined
        label_combined_dir = os.path.join(save_dir, label)
        os.makedirs(label_combined_dir, exist_ok=True)

        for img_name in os.listdir(label_path):
            try:
                region_imgs = []
                for region in regions:
                    img_path = os.path.join(base_dir, set_name, region, label, img_name)
                    img = Image.open(img_path).resize((128, 128))
                    region_imgs.append(img)

                # Ghép theo chiều dọc
                total_height = 128 * len(region_imgs)
                combined_img = Image.new("RGB", (128, total_height))
                for i, img in enumerate(region_imgs):
                    combined_img.paste(img, (0, i * 128))

                save_path = os.path.join(label_combined_dir, img_name)
                combined_img.save(save_path)
            except Exception as e:
                print(f"❌ Error processing {img_name}: {e}")
