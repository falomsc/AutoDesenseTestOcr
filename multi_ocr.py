import cv2

from ocr_utility import get_data_from_pic2


def image_processing(img_path: str):
    enhanced_image = cv2.convertScaleAbs(cv2.imread(img_path), alpha=1.4, beta=0)
    return enhanced_image

for i in range(20):
    img_path = f"result/920063/ppg/screenshot_{i}.png"
    res = get_data_from_pic2(img=img_path, image_processing=image_processing)
    for k, v in res.items():
        print(k, v)
    print("-" * 30)