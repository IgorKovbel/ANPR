import math
import time
import numpy as np
import torch
import cv2


def calculate_fps(prev_time, fps_buffer, buffer_size):
    current_time = time.time()
    time_diff = current_time - prev_time
    if time_diff == 0:
        fps = 0
    else:
        fps = 1.0 / time_diff
    fps_buffer.append(fps)
    if len(fps_buffer) > buffer_size:
        fps_buffer.pop(0)
    avg_fps = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0
    return avg_fps, current_time


def smooth_coordinates(current_coords, prev_coords, frame_width, frame_height, alpha=0.2):
    smoothed_coords = []
    for cur, prev in zip(current_coords, prev_coords):
        smoothed_value = alpha * cur + (1 - alpha) * prev
        
        if smoothed_value < 0:
            smoothed_value = 0
        elif smoothed_value > frame_width and (len(smoothed_coords) % 2 == 0):
            smoothed_value = frame_width
        elif smoothed_value > frame_height and (len(smoothed_coords) % 2 == 1):
            smoothed_value = frame_height
        
        smoothed_coords.append(smoothed_value)
    
    return smoothed_coords


def select_best_box_and_platenumber(detections, center_screen_x, center_screen_y):
    best_car_box = None
    best_car_box_score = 0.0
    best_distance = float("inf")
    
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection

        if int(class_id) == 1:
            if x1 <= center_screen_x <= x2 and y1 <= center_screen_y <= y2:

                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2

                distance = math.hypot(box_center_x - center_screen_x,
                                      box_center_y - center_screen_y)

                if distance < best_distance:
                    best_distance = distance
                    best_car_box = [x1, y1, x2, y2]
                    best_car_box_score = score

    if best_car_box is None:
        return None, 0.0, None, 0.0

    licence_number_box = None
    licence_score = 0.0

    car_x1, car_y1, car_x2, car_y2 = best_car_box
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection

        if int(class_id) == 0:
            if x1 >= car_x1 and y1 >= car_y1 and x2 <= car_x2 and y2 <= car_y2:
                if score > licence_score:
                    licence_score = score
                    licence_number_box = [x1, y1, x2, y2]

    return best_car_box, best_car_box_score, licence_number_box, licence_score


def draw_center_crosshair(img, width, height, size=20, color=(192,192,192), alpha=0.5):
    center_x = width // 2
    center_y = height // 2
    
    overlay = img.copy()
    
    cv2.line(overlay, (center_x - size, center_y),
             (center_x + size, center_y), color, 2)

    cv2.line(overlay, (center_x, center_y - size),
             (center_x, center_y + size), color, 2)
    

    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)