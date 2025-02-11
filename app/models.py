# app/models.py
import onnx
import onnxruntime
import torch
from libs.fast.fast_plate_ocr import ONNXPlateRecognizer
from libs.yolov10.ultralytics import YOLOv10
from libs.HiT.tracking.video_demo import HiT

ocr_model = ONNXPlateRecognizer('european-plates-mobile-vit-v2-model')

coco_model = YOLOv10("app\models\model.pt")

net_path = 'app\models\VT_ep1500.onnx'
onnx_model = onnx.load(net_path)
onnx.checker.check_model(onnx_model)
with torch.no_grad():
    ort_session = onnxruntime.InferenceSession(
        net_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    print("ONNX Runtime device:", onnxruntime.get_device())

car_tracker = HiT(ort_session)
license_tracker = HiT(ort_session)
