# app/main.py
import asyncio
import uuid
import time
import os
import io
import cv2

from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from aiortc import RTCPeerConnection, RTCSessionDescription

from app.processing import process_image_car_and_plate, process_video_like_webrtc
from app.webrtc import OpenCvVideoTrack, latest_license_plate

app = FastAPI()
templates = Jinja2Templates(directory="templates")

pcs = {}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/offer")
async def offer(request: Request):
    data = await request.json()
    offer_sdp = data.get("sdp")
    offer_type = data.get("type")
    if not offer_sdp or not offer_type:
        raise HTTPException(status_code=400, detail="Missing SDP data")

    pc = RTCPeerConnection()
    pc_id = str(uuid.uuid4())
    pcs[pc_id] = pc

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            local_video = OpenCvVideoTrack(track)
            pc.addTrack(local_video)

    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type=offer_type))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

@app.post("/stop")
async def stop():
    coros = [pc.close() for pc in pcs.values()]
    await asyncio.gather(*coros, return_exceptions=True)
    pcs.clear()
    return {"message": "All connections closed"}

@app.get("/latest_plate")
async def latest_plate_endpoint():
    return {"plate": latest_license_plate}

@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        processed_img, plate = process_image_car_and_plate(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Fail image processing: {e}")

    ret, encoded_img = cv2.imencode(".jpg", processed_img)
    img_bytes = encoded_img.tobytes()

    headers = {"X-Recognized-Plate": plate if plate else "NOT_FOUND"}
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg", headers=headers)

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        processed_video_bytes = process_video_like_webrtc(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Fail video processing: {e}")
    return StreamingResponse(io.BytesIO(processed_video_bytes), media_type="video/mp4")
