import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from multiprocessing import Queue
from typing import Callable, Any

import numpy as np
from paddleocr import PaddleOCR

from zepp import Zepp


def get_data_from_pic1(ocr: PaddleOCR, img, is_gps_enabled: bool = True, is_bds_enabled: bool = True,
                       is_gal_enabled: bool = True, is_gln_enabled: bool = True, is_utc_time_enabled: bool = True,
                       is_top4_cn_enabled: bool = True, is_pos_ttff_enabled: bool = True,
                       last_utc_time: datetime = None):
    """
    一次遍历的方法，可能会有问题
    :param ocr:
    :param img:
    :param is_gps_enabled:
    :param is_bds_enabled:
    :param is_gal_enabled:
    :param is_gln_enabled:
    :param is_utc_time_enabled:
    :param is_top4_cn_enabled:
    :param is_pos_ttff_enabled:
    :param last_utc_time: if current_utc_time == last_utc_time, skip.
    :return:
    """
    result = ocr.ocr(img, cls=False)
    utc_line_countdown = 0
    gnss_l1_countdown = 0
    gnss_data_dict = dict()
    gnss_text_dict = dict()

    for data in result[0]:
        text = data[1][0]
        pos_list = data[0]
        current_pos_y = get_center(pos_list)[1]

        # utc time
        if is_utc_time_enabled and re.match('utc time', text):
            utc_line_countdown = 1
            continue
        if utc_line_countdown and is_utc_time_enabled:
            current_utc_time = datetime.strptime(re.search(r'\d\d:\d\d:\d\d', text).group(0), "%H:%M:%S").time()
            if not last_utc_time or last_utc_time != current_utc_time:
                gnss_data_dict['utc_time'] = current_utc_time
            else:
                break
            utc_line_countdown -= 1
            is_utc_time_enabled = False

        # top4 cn
        if is_top4_cn_enabled and re.match('t[0Oo]p4_cn', text):
            gnss_data_dict['top4_cn'] = float(re.search(r'\d+\.\d', text).group(0))
            is_top4_cn_enabled = False

        # pos ttff
        if is_pos_ttff_enabled and re.match('p[0Oo]s_tt?ff', text):
            gnss_data_dict['pos_ttff'] = int(re.search(r'\d+', text).group(0))
            is_pos_ttff_enabled = False

        # gnss data
        if gnss_l1_countdown:
            if l1_pos_y is not None:
                if abs(l1_pos_y - current_pos_y) < 5:
                    l1_text += text
                else:
                    gnss_text_dict[gnss_typ].append(l1_text)
                    l1_text = ''
                    gnss_l1_countdown -= 1
            if re.match(r'L1', text):
                l1_pos_y = current_pos_y
                l1_text = text
                continue

        # gnss c/n
        if is_gps_enabled and re.match('gps_gsv_info', text):
            gnss_l1_countdown = 4
            gnss_typ = "gps"
            gnss_data_dict["gps"] = dict()
            gnss_text_dict["gps"] = list()
            is_gps_enabled = False
            l1_pos_y = None
            continue

        if is_bds_enabled and re.match('bds_gsv_info', text):
            gnss_l1_countdown = 4
            gnss_typ = "bds"
            gnss_data_dict["bds"] = dict()
            gnss_text_dict["bds"] = list()
            is_bds_enabled = False
            l1_pos_y = None
            continue

        if is_gal_enabled and re.match('gal_gsv_info', text):
            gnss_l1_countdown = 4
            gnss_typ = "gal"
            gnss_data_dict["gal"] = dict()
            gnss_text_dict["gal"] = list()
            is_gal_enabled = False
            l1_pos_y = None
            continue

        if is_gln_enabled and re.match('gln_gsv_info', text):
            gnss_l1_countdown = 4
            gnss_typ = "gln"
            gnss_data_dict["gln"] = dict()
            gnss_text_dict["gln"] = list()
            is_gln_enabled = False
            l1_pos_y = None
            continue

    for gnss_typ, lines in gnss_text_dict.items():
        for line in lines:
            line = line.translate(str.maketrans('Ol', '01'))
            match = re.search(r"(\d{3})\s?(\d{2})\.?(\d)", line)
            gnss_data_dict[gnss_typ].update({match.group(1): float(f"{match.group(2)}.{match.group(3)}")})

    return gnss_data_dict


def get_data_from_pic2(ocr: PaddleOCR, img: str, gnss_typ: str = "gps", is_utc_time_enabled: bool = True,
                       is_top4_cn_enabled: bool = True, is_pos_ttff_enabled: bool = True,
                       last_utc_time: datetime = None, image_processing: Callable[[str], Any] | None = None):
    """
    两次遍历的方法
    一次只能测一种制式
    :param ocr:
    :param img:
    :param gnss_typ: "gps", "bds", "gal", "gln"
    :param is_utc_time_enabled:
    :param is_top4_cn_enabled:
    :param is_pos_ttff_enabled:
    :param last_utc_time: if current_utc_time == last_utc_time, skip.
    :param image_processing:
    :return:
    """
    if image_processing:
        img = image_processing(img)
    result = ocr.ocr(img, cls=False)
    utc_time_pos_y = None
    top4_cn_pos_y = None
    pos_ttff_pos_y = None
    is_gnss_title_find = False
    gnss_l1_num = 0
    gnss_l1_pos_y_dict = dict()
    gnss_title_dict = {"gps": "gps_gsv_info", "bds": "bds_gsv_info", "gal": "gal_gsv_info", "gln": "gal_gsv_info"}
    '''
    gnss_l1_pos_y_dict = {"gps": [l1_pos_y1, l1_pos_y2, l1_pos_y3, l1_pos_y4]}
    '''
    gnss_data_dict = dict()
    gnss_text_dict = dict()
    '''
    gnss_text_dict = {"gps": [[("L1", pos_x_00), ("37.5", pos_x_01), ("012", pos_x_02)],
                              [("L1013385", pos_x_10),],
                              [("L1", pos_x_20), ("01439.5", pos_x_21)],
                              [("L1", pos_x_30), ("015", pos_x_31), ("40.5", pos_x_32)]]}
    '''

    for data in result[0]:
        text = data[1][0]
        pos_list = data[0]
        current_pos_y = get_center(pos_list)[1]

        # utc time pos y
        if is_utc_time_enabled and re.match('utc time', text):
            utc_time_pos_y = current_pos_y
            is_utc_time_enabled = False
            continue
        # top4 cn pos y
        elif is_top4_cn_enabled and re.match('t[0Oo]p4_cn', text):
            top4_cn_pos_y = current_pos_y
            is_top4_cn_enabled = False
            continue
        # pos ttff pos y
        elif is_pos_ttff_enabled and re.match("p[0Oo]s_tt?ff", text):
            pos_ttff_pos_y = current_pos_y
            is_pos_ttff_enabled = False
            continue

        # gnss pos y
        elif (not is_gnss_title_find) and re.match(gnss_title_dict[gnss_typ], text):
            gnss_l1_pos_y_dict[gnss_typ] = list()
            gnss_text_dict[gnss_typ] = [[] for _ in range(4)]
            gnss_data_dict[gnss_typ] = dict()
            is_gnss_title_find = True
            continue

        # L1 pos y
        elif is_gnss_title_find and re.match(r'L[1l]', text) and gnss_l1_num < 4:
            gnss_l1_pos_y_dict[gnss_typ].append(current_pos_y)
            gnss_l1_num += 1
            continue

    for data in result[0]:
        text = data[1][0]
        pos_list = data[0]
        current_pos_x = get_center(pos_list)[0]
        current_pos_y = get_center(pos_list)[1]

        # find utc_time
        if (abs(utc_time_pos_y - current_pos_y) < 5) and (utc_time_re := re.search(r'\d\d:\d\d:\d\d', text)):
            utc_time = datetime.strptime(utc_time_re.group(0), "%H:%M:%S").time()
            if not last_utc_time or last_utc_time != utc_time:
                gnss_data_dict["utc_time"] = utc_time
                continue
            else:
                return
        # find top4_cn
        elif (abs(top4_cn_pos_y - current_pos_y) < 5) and (top4_cn_re := re.search(r"(\d\d)\.?(\d)", text)):
            top4_cn = float(f"{top4_cn_re.group(1)}.{top4_cn_re.group(2)}")
            gnss_data_dict["top4_cn"] = top4_cn
            continue
        # find pos ttff
        elif (abs(pos_ttff_pos_y - current_pos_y) < 5) and (pos_ttff_re := re.search(r'[\dO]{1,4}$', text.strip())):
            pos_ttff = int(pos_ttff_re.group(0).replace("O", "0"))
            gnss_data_dict["pos_ttff"] = pos_ttff
            continue

        # find gnss cn
        if gnss_l1_pos_y_dict.get(gnss_typ):
            for i in range(4):
                if abs(gnss_l1_pos_y_dict[gnss_typ][i] - current_pos_y) < 5:
                    gnss_text_dict[gnss_typ][i].append((text, current_pos_x))
                    continue

    for cells in gnss_text_dict[gnss_typ]:
        sorted_cells = sorted(cells, key=lambda x: x[1])
        l1_text = (''.join(item[0] for item in sorted_cells)).translate(str.maketrans('Ol', '01')).strip()
        match = re.search(r"(\d{3})\s?(\d)\2?(\d)\.?(\d)$", l1_text)
        gnss_data_dict[gnss_typ].update({match.group(1): float(f"{match.group(2)}{match.group(3)}.{match.group(4)}")})

    return gnss_data_dict


def get_center(pos_list: dict) -> np.ndarray:
    return np.sum(pos_list, axis=0) / 4


def get_min_top4cn(res_list: list, gnss_typ: str, cn_max_diff: int):
    min_entry = None
    min_top4cn = float('inf')
    for entry in res_list:
        gps_data = entry.get(gnss_typ, {})
        if len(gps_data) < 4:
            continue
        top_cn = entry.get('top4_cn', None)
        if top_cn is None:
            continue
        gnss_cn_values = list(gps_data.values())
        if max(gnss_cn_values) - min(gnss_cn_values) < cn_max_diff:
            if top_cn < min_top4cn:
                min_top4cn = top_cn
                min_entry = entry
    return min_entry


def take_screenshot(filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        result = subprocess.run(['adb', 'exec-out', 'screencap', '-p'], stdout=f)
        if result.returncode == 0:
            print(f'Saved: {filename}')
        else:
            print(f'Error taking screenshot: {result.returncode}')


def take_screenshots(screenshot_interval: float, duration: int, filename: str, queue: Queue = None,
                     case_name: str = None):
    for i in range(int(duration / screenshot_interval)):
        current_time = time.time()
        take_screenshot(f"{filename}_{i}.png")
        if queue:
            queue.put((case_name, f"{filename}_{i}.png"))
        elapsed_time = time.time() - current_time
        sleep_time = max(0, screenshot_interval - elapsed_time)
        time.sleep(sleep_time)


def delete_all_files(filename: str):
    directory = filename if os.path.isdir(filename) else os.path.dirname(filename)
    if not os.path.exists(directory):
        print(f"{directory} not exists")
        return
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def gnss_positioning1(zepp: Zepp, ocr: PaddleOCR, ttff_scan_time: int, img: str = "./result/tmp/screenshot.png",
                      image_processing: Callable[[str], Any] | None = None):
    zepp.send("hm:gnss+proc=open")
    zepp.send("hm:gnss+proc=read_auto")
    while (1):
        time.sleep(ttff_scan_time)
        take_screenshot(img)
        res = get_data_from_pic2(ocr=ocr, img=img, image_processing=image_processing)
        if res['pos_ttff'] > 0:
            break


def gnss_positioning2(zepp: Zepp, img_queue, ttff_queue, ttff_scan_time: int, ttff_img_path: str):
    zepp.send("hm:gnss+proc=open")
    zepp.send("hm:gnss+proc=read_auto")
    while(1):
        time.sleep(ttff_scan_time)
        take_screenshot(ttff_img_path)
        img_queue.put("positioning")
        res = ttff_queue.get()
        if res['pos_ttff'] > 0:
            break
