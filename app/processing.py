# app/processing.py
import os
import cv2
import time
import numpy as np
import io

from app.models import coco_model, ocr_model, ort_session
from app.utils import select_best_box_and_platenumber, draw_center_crosshair, smooth_coordinates
from libs.HiT.tracking.video_demo import HiT

def process_image_car_and_plate(file_bytes: bytes):

    file_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Error")

    height, width = img.shape[:2]

    results = coco_model(img)[0] 

    car_box, car_score, lic_box, lic_score = select_best_box_and_platenumber(results, width / 2, height / 2)

    recognized_plate = None

    if car_box and car_score > 0.5:
        x1, y1, x2, y2 = map(int, car_box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

    if lic_box and lic_score > 0.5:
        x1, y1, x2, y2 = map(int, lic_box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        plate_roi = img[y1:y2, x1:x2]
        gray_crop = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        plate_data = ocr_model.run(gray_crop, return_confidence=True)
        plate_text = plate_data[0][0].replace('_', '')
        confs = plate_data[1][0][:len(plate_text)]
        if np.all(np.array(confs) > 0.5):
            recognized_plate = plate_text
            text_size = cv2.getTextSize(recognized_plate, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
            cv2.rectangle(img, (x1 - 15, y1),
                          (x1 - 15 + text_size[0] + 10, y1 - 15 - text_size[1] - 10),
                          (0, 0, 0), -1)
            cv2.putText(img, recognized_plate, (x1 - 15, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    img = draw_center_crosshair(img, width, height, size=20, color=(192, 192, 192), alpha=0.5)
    return img, recognized_plate

def draw_video_overlays(
    frame, width, height,
    car_detected, car_box,
    licence_detected, lic_box,
    licence_recognized, licence_plate,
    previous_coords
):

    if car_detected and car_box is not None:
        if previous_coords['car'] is not None:
            smoothed = smooth_coordinates(list(car_box), previous_coords['car'], width, height, alpha=0.2)
            previous_coords['car'] = smoothed
        else:
            previous_coords['car'] = list(car_box)
        x1, y1, x2, y2 = map(int, previous_coords['car'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    if licence_detected and lic_box is not None:
        if previous_coords['licence'] is not None:
            smoothed = smooth_coordinates(list(lic_box), previous_coords['licence'], width, height, alpha=0.7)
            previous_coords['licence'] = smoothed
        else:
            previous_coords['licence'] = list(lic_box)
        x1, y1, x2, y2 = map(int, previous_coords['licence'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        if licence_recognized and licence_plate:
            text_size = cv2.getTextSize(licence_plate, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
            cv2.rectangle(frame, (x1 - 15, y1),
                          (x1 - 15 + text_size[0] + 10, y1 - 15 - text_size[1] - 10),
                          (0, 0, 0), -1)
            cv2.putText(frame, licence_plate, (x1 - 15, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    frame = draw_center_crosshair(frame, width, height, size=20, color=(192, 192, 192), alpha=0.5)
    return frame

def process_video_like_webrtc(file_bytes: bytes):
    in_filename = "temp_input.mp4"
    with open(in_filename, "wb") as f:
        f.write(file_bytes)

    cap = cv2.VideoCapture(in_filename)
    if not cap.isOpened():
        os.remove(in_filename)
        raise ValueError("Error")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_filename = "temp_output.mp4"
    out = cv2.VideoWriter(out_filename, fourcc, fps, (width, height))

    frame_nmr = -1
    car_detected = False
    licence_detected = False
    licence_recognized = False
    licence_plate = ""
    last_car_box = None
    last_lic_box = None
    previous_coords = {'car': None, 'licence': None}

    local_car_tracker = HiT(ort_session)
    local_licence_tracker = HiT(ort_session)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_nmr += 1

        if frame_nmr % 15 == 0:
            detections = coco_model(frame)[0]
            car_box, car_score, lic_box, lic_score = select_best_box_and_platenumber(
                detections, width / 2, height / 2
            )
            if car_box and car_score > 0.5:
                x1, y1, x2, y2 = map(int, car_box)
                local_car_tracker.initialize(frame, {'init_bbox': [x1, y1, x2 - x1, y2 - y1]})
                car_detected = True
                last_car_box = (x1, y1, x2, y2)
            else:
                car_detected = False
                last_car_box = None

            if lic_box and lic_score > 0.5:
                x1, y1, x2, y2 = map(int, lic_box)
                local_licence_tracker.initialize(frame, {'init_bbox': [x1, y1, x2 - x1, y2 - y1]})
                licence_detected = True
                last_lic_box = (x1, y1, x2, y2)
                crop = frame[y1:y2, x1:x2]
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                plate_data = ocr_model.run(gray_crop, return_confidence=True)
                plate_text = plate_data[0][0].replace('_', '')
                conf_array = plate_data[1][0][:len(plate_text)]
                if np.all(np.array(conf_array) > 0.5):
                    licence_plate = plate_text
                    licence_recognized = True
                else:
                    licence_plate = ""
                    licence_recognized = False
            else:
                licence_detected = False
                last_lic_box = None
                licence_recognized = False
                licence_plate = ""
        else:
            if car_detected and last_car_box is not None:
                car_out = local_car_tracker.track(frame)
                x1, y1, w, h = map(int, car_out['target_bbox'])
                last_car_box = (x1, y1, x1 + w, y1 + h)
            if licence_detected and last_lic_box is not None:
                lic_out = local_licence_tracker.track(frame)
                x1, y1, w, h = map(int, lic_out['target_bbox'])
                last_lic_box = (x1, y1, x1 + w, y1 + h)

        frame = draw_video_overlays(
            frame, width, height,
            car_detected, last_car_box,
            licence_detected, last_lic_box,
            licence_recognized, licence_plate,
            previous_coords
        )
        out.write(frame)

    cap.release()
    out.release()
    os.remove(in_filename)
    with open(out_filename, "rb") as f:
        processed_video_bytes = f.read()
    #os.remove(out_filename)
    return processed_video_bytes
