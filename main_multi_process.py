import logging
import time
from datetime import datetime
from multiprocessing import Queue, Process

import cv2
import yaml
from paddleocr import PaddleOCR

from ocr_utility import delete_all_files, take_screenshots, get_data_from_pic2, get_min_top4cn, gnss_positioning2
from zepp import Zepp


def image_processing(img_path: str):
    enhanced_image = cv2.convertScaleAbs(cv2.imread(img_path), alpha=1.4, beta=0)
    return enhanced_image


def screenshots_process(img_queue, ttff_queue, mobile_sn, input_resource_id, send_resource_id, testing_cases, dut_sn,
                        screenshot_interval, reboot_time, ttff_scan_time, case_interval, commands, ttff_img_path):
    zepp = Zepp(mobile_sn=mobile_sn, input_resource_id=input_resource_id, send_resource_id=send_resource_id)

    gnss_positioning2(zepp, img_queue, ttff_queue, ttff_scan_time, ttff_img_path)

    for case_name, case_settings in testing_cases.items():
        print(case_name, "start")
        for step in case_settings[0]:
            if type(step) == int:
                time.sleep(step)
            elif step == "testing":
                delete_all_files(filename=f"result/{dut_sn}/{case_name}/")
                take_screenshots(screenshot_interval=screenshot_interval, duration=case_settings[1],
                                 filename=f"result/{dut_sn}/{case_name}/screenshot", queue=img_queue, case_name=case_name)
            elif step == "reboot":
                zepp.send(commands["reboot"])
                time.sleep(reboot_time)
                gnss_positioning2(zepp, img_queue, ttff_queue, ttff_scan_time, ttff_img_path)
            else:
                zepp.send(commands[step])
        time.sleep(case_interval)
    img_queue.put("end")


if __name__ == '__main__':
    log_filename = datetime.now().strftime("logfile_%Y-%m-%d_%H-%M-%S.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filename, mode='w')
        ]
    )
    logger = logging.getLogger('myLogger')


    with open("config/zepp.yaml") as f1, open("config/testing.yaml") as f2:
        zepp_config_dict = yaml.load(f1, Loader=yaml.FullLoader)
        testing_config_dict = yaml.load(f2, Loader=yaml.FullLoader)
    dut_sn = testing_config_dict["dut_sn"]
    gnss_typ = testing_config_dict["gnss_typ"]
    ttff_scan_time = testing_config_dict["ttff_scan_time"]
    cn_max_diff = testing_config_dict["cn_max_diff"]
    testing_cases = testing_config_dict['testing_cases']
    screenshot_interval = testing_config_dict["screenshot_interval"]
    reboot_time = testing_config_dict["reboot_time"]
    case_interval = testing_config_dict["case_interval"]

    mobile_sn = zepp_config_dict["mobile_sn"]
    input_resource_id = zepp_config_dict["input_resource_id"]
    send_resource_id = zepp_config_dict["send_resource_id"]
    commands = zepp_config_dict["commands"]

    img_queue = Queue(maxsize=1000)
    ttff_queue = Queue(maxsize=1000)

    last_utc_time_dict = {}  # {testing_case1: utc_time1, testing_case2: utc_time2, ...}
    ocr_res_dict = {}  # {testing_case1: [ocr_data_dict1_1, ocr_data_dict1_2, ...], testing_case2: [ocr_data_dict2_1, ocr_data_dict2_2, ...], ...}
    min_entry = 0
    min_res_dict = {}  # {testing_case1: ocr_data_dict1_min, testing_case2: ocr_data_dict2_min, ...}
    ttff_img_path = "./result/tmp/screenshot.png"

    p_capture = Process(target=screenshots_process, args=(
        img_queue, mobile_sn, input_resource_id, send_resource_id, testing_cases, dut_sn, screenshot_interval, reboot_time,
        ttff_scan_time, case_interval, commands, ttff_img_path))

    ocr = PaddleOCR(lang='en', det_limit_side_len=4000)
    p_capture.start()

    while True:
        queue_data = img_queue.get()
        if queue_data == "end":
            break
        elif queue_data == "positioning":
            res = get_data_from_pic2(ocr=ocr, img=ttff_img_path, image_processing=image_processing)

        testing_case, img_path = queue_data
        last_utc_time_dict.setdefault(testing_case, None)
        ocr_data_dict = get_data_from_pic2(
            ocr=ocr,
            img=img_path,
            gnss_typ=gnss_typ,
            last_utc_time=last_utc_time_dict[testing_case],
            image_processing=image_processing)
        if ocr_data_dict:
            last_utc_time_dict[testing_case] = ocr_data_dict.get('utc_time')
            ocr_res_dict.setdefault(testing_case, []).append(ocr_data_dict)

    for testing_case, ocr_data_list in ocr_res_dict.items():
        min_entry = get_min_top4cn(res_list=ocr_data_list, gnss_typ=gnss_typ, cn_max_diff=cn_max_diff)
        min_res_dict.update({testing_case: min_entry})

    print("-" * 50)
    for k, v in ocr_res_dict.items():
        print(f"{k}: {v}")
    print("*" * 50)
    for k, v in min_res_dict.items():
        print(f"{k}: {v}")
