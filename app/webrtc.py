# app/webrtc.py
import asyncio
import time
import cv2
import numpy as np
from aiortc import VideoStreamTrack
from aiortc.mediastreams import VideoFrame

from app.models import coco_model, ocr_model, car_tracker, license_tracker, ort_session
from app.utils import select_best_box_and_platenumber, draw_center_crosshair, smooth_coordinates

latest_license_plate = ""

class OpenCvVideoTrack(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()
        self.track = track
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.frame_nmr = -1

        self.car_detected = False
        self.licence_detected = False
        self.licence_recognized = False
        self.licence_plate = ""
        self.conf = []

        self.last_car_box = None
        self.last_licence_box = None

        self.previous_coords = {'car': None, 'licence': None}

        self.width = None
        self.height = None

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        if self.width is None or self.height is None:
            self.height, self.width = img.shape[:2]
            print(f"Frame size: {self.width}x{self.height}")
        else:
            new_height, new_width = img.shape[:2]
            if new_width != self.width or new_height != self.height:
                self.width, self.height = new_width, new_height
                print(f"Frame size changed to: {self.width}x{self.height}")

        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1:
            self.fps = self.frame_count / elapsed_time
            self.start_time = time.time()
            self.frame_count = 0

        self.frame_nmr += 1

        if self.frame_nmr % 15 == 0:
            detections = await asyncio.to_thread(lambda: coco_model(img)[0])
            await self.handle_detections(detections, img)
        else:
            if self.car_detected and self.last_car_box is not None:
                self.update_car_tracker(img)
            if self.licence_detected and self.last_licence_box is not None:
                self.update_licence_tracker(img)

        img = self.draw_overlays(img)
        cv2.putText(img, f"FPS: {self.fps:.2f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

    async def handle_detections(self, detections, img):
        global latest_license_plate
        car_box, car_score, lic_box, lic_score = select_best_box_and_platenumber(
            detections, self.width / 2, self.height / 2
        )
        if car_box and car_score > 0.5:
            x1, y1, x2, y2 = map(int, car_box)
            car_tracker.initialize(img, {'init_bbox': [x1, y1, x2 - x1, y2 - y1]})
            self.car_detected = True
            self.last_car_box = (x1, y1, x2, y2)
        else:
            self.car_detected = False
            self.last_car_box = None

        if lic_box and lic_score > 0.5:
            x1, y1, x2, y2 = map(int, lic_box)
            license_tracker.initialize(img, {'init_bbox': [x1, y1, x2 - x1, y2 - y1]})
            self.licence_detected = True
            self.last_licence_box = (x1, y1, x2, y2)
            crop = img[y1:y2, x1:x2]
            plate_data = await asyncio.to_thread(
                lambda: ocr_model.run(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), return_confidence=True)
            )
            self.licence_plate = plate_data[0][0].replace('_', '')
            self.conf = plate_data[1][0][:len(self.licence_plate)]
            if np.all(np.array(self.conf) > 0.5):
                self.licence_recognized = True
                latest_license_plate = self.licence_plate
            else:
                self.licence_recognized = False
        else:
            self.licence_detected = False
            self.last_licence_box = None
            self.licence_recognized = False
            self.licence_plate = ""

    def update_car_tracker(self, img):
        car_out = car_tracker.track(img)
        x1, y1, w, h = map(int, car_out['target_bbox'])
        self.last_car_box = (x1, y1, x1 + w, y1 + h)

    def update_licence_tracker(self, img):
        lic_out = license_tracker.track(img)
        x1, y1, w, h = map(int, lic_out['target_bbox'])
        self.last_licence_box = (x1, y1, x1 + w, y1 + h)

    def draw_overlays(self, img):
        if self.car_detected and self.last_car_box is not None:
            if self.previous_coords['car'] is not None:
                self.previous_coords['car'] = smooth_coordinates(
                    list(self.last_car_box), self.previous_coords['car'], self.width, self.height, alpha=0.2
                )
            else:
                self.previous_coords['car'] = list(self.last_car_box)
            x1, y1, x2, y2 = map(int, self.previous_coords['car'])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 5)

        if self.licence_detected and self.last_licence_box is not None:
            if self.previous_coords['licence'] is not None:
                self.previous_coords['licence'] = smooth_coordinates(
                    list(self.last_licence_box), self.previous_coords['licence'], self.width, self.height, alpha=0.7
                )
            else:
                self.previous_coords['licence'] = list(self.last_licence_box)
            x1, y1, x2, y2 = map(int, self.previous_coords['licence'])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 5)
            if self.licence_recognized:
                text_size = cv2.getTextSize(self.licence_plate, cv2.FONT_HERSHEY_SIMPLEX, 1, 3)[0]
                cv2.rectangle(img, (x1 - 15, y1),
                              (x1 - 15 + text_size[0] + 10, y1 - 15 - text_size[1] - 10),
                              (0, 0, 0), -1)
                cv2.putText(img, self.licence_plate, (x1 - 15, y1 - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        img = draw_center_crosshair(img, self.width, self.height, size=20, color=(192, 192, 192), alpha=0.5)
        return img
