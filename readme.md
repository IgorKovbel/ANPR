# ANPR
<video controls>
  <source src='./result.mp4' type="video/mp4">
  Your browser does not support the video tag.
</video>

# Licence Plate Project

in this project I used a popular YOLOv10 detector, and to improve fps I also added a visual object tracker - HiT, which allows you to process video using cuda in 60 + fps and also allows you to comfortably apply in real time, without lags. An ONNX-based OCR model to recognize numbers. In addition, there are APIs for image and video processing, as well as the ability to work in real time WebRTC. But my implementation is highly dependent on the virtual environment, library versions, etc., so it may not work stably. I trained a model on an assembly of various datasets with cars and license plates and a significant part of the white dataset is taken from Auto.ria

# Features

* Image processing: image loading, car and license plate detection, OCR for number recognition.
* Video processing: download video with frame-by-frame processing (detection every 15 frames, HiT tracking) and return the finished mp4 file.
* WebRTC: processing the video stream from the camera in real time.


## Links
* OCR - <a href='https://github.com/ankandrew/fast-plate-ocr'>Fast & Lightweight License Plate OCR</a>
* Visual Object Tracker - <a href='https://github.com/kangben258/hit'>Exploring Lightweight Hierarchical Vision Transformers for Efficient Visual Tracking</a>
* Object Detector - <a href='https://github.com/THU-MIG/yolov10'>YOLOv10</a>

## My talk

Ideally, the project should be done on the edge device, because although WebRTC is quite fast, there is still a fairly significant delay, which still depends on the distance between the server and the client. But I am still satisfied with the result.