from paddleocr import PaddleOCR
import cv2

from ocr_utility import get_data_from_pic1, get_data_from_pic2

def image_processing(img_path: str):
    enhanced_image = cv2.convertScaleAbs(cv2.imread(img_path), alpha=1.4, beta=0)
    return enhanced_image


img_path = "./result/tmp/screenshot.png"

# ocr = PaddleOCR(lang='en')
ocr = PaddleOCR(lang='en', det_limit_side_len=3500)

result = ocr.ocr(image_processing(img_path), cls=False)
for data in result[0]:
    text = data[1][0]
    print(text)

print("-" * 30)
res = get_data_from_pic2(ocr=ocr, img=img_path, image_processing=image_processing)

for k, v in res.items():
    print(k, v)