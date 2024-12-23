import time

import cv2
import yaml

from paddleocr import PaddleOCR

from ocr_utility import take_screenshots, get_data_from_pic2, get_min_top4cn, delete_all_files, gnss_positioning1
from zepp import Zepp


def image_processing(img_path: str):
    enhanced_image = cv2.convertScaleAbs(cv2.imread(img_path), alpha=1.4, beta=0)
    return enhanced_image


ocr = PaddleOCR(lang='en', det_limit_side_len=4000)

with open("config/zepp.yaml") as f1, open("config/testing.yaml") as f2:
    zepp_config_dict = yaml.load(f1, Loader=yaml.FullLoader)
    testing_config_dict = yaml.load(f2, Loader=yaml.FullLoader)

start = time.time()

sn = testing_config_dict["sn"]
gnss_typ = testing_config_dict["gnss_typ"]
ttff_scan_time = testing_config_dict["ttff_scan_time"]
cn_max_diff = testing_config_dict["cn_max_diff"]
reboot_time = testing_config_dict["reboot_time"]
screenshot_interval = testing_config_dict["screenshot_interval"]
case_interval = testing_config_dict["case_interval"]
mobile_sn = zepp_config_dict["mobile_sn"]
input_resource_id = zepp_config_dict["input_resource_id"]
send_resource_id = zepp_config_dict["send_resource_id"]
last_utc_time = None
ocr_res_list = []
min_entry = 0
testing_res_dict = {}

zepp = Zepp(mobile_sn=mobile_sn, input_resource_id=input_resource_id, send_resource_id=send_resource_id)
gnss_positioning1(zepp=zepp, ocr=ocr, ttff_scan_time=ttff_scan_time)

for testing_case, case_setting in testing_config_dict['testing_cases'].items():
    last_step = None
    print(testing_case, "start")
    for step in case_setting[0]:
        if type(step) == int:
            time.sleep(step)
        elif step == "testing":
            delete_all_files(filename=f"result/{sn}/{testing_case}/")
            take_screenshots(screenshot_interval=screenshot_interval, duration=case_setting[1],
                             filename=f"result/{sn}/{testing_case}/screenshot")
        elif step == "reboot":
            last_step = "reboot"
            zepp.send(zepp_config_dict["commands"]["reboot"])
        else:
            zepp.send(zepp_config_dict["commands"][step])
    case_end_time = time.time()

    for i in range(int(case_setting[1] / screenshot_interval)):
        img_path = f"result/{sn}/{testing_case}/screenshot_{i}.png"
        print(f"get_data_from: {img_path}")
        ocr_res = get_data_from_pic2(ocr=ocr, img=img_path, gnss_typ=gnss_typ, last_utc_time=last_utc_time,
                                     image_processing=image_processing)
        if ocr_res:
            last_utc_time = ocr_res.get('utc_time')
            ocr_res_list.append(ocr_res)
    min_entry = get_min_top4cn(res_list=ocr_res_list, gnss_typ=gnss_typ, cn_max_diff=cn_max_diff)
    testing_res_dict.update({testing_case: min_entry})
    print(testing_case, min_entry)

    data_analysis_end_time = time.time()
    if last_step == "reboot":
        if data_analysis_end_time - case_end_time < reboot_time:
            time.sleep(reboot_time - (data_analysis_end_time - case_end_time))
        gnss_positioning1(zepp=zepp, ocr=ocr, ttff_scan_time=ttff_scan_time, image_processing=image_processing)
    else:
        if data_analysis_end_time - case_end_time < case_interval:
            time.sleep(case_interval - (data_analysis_end_time - case_end_time))

print(testing_res_dict)
stop = time.time()
print(stop - start)
