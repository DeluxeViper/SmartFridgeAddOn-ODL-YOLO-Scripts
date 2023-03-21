import cv2
import depthai as dai
import time
from pathlib import Path
from PIL import Image
import numpy as np

blue = cv2.imread("blue.jpeg")

# Create pipeline
pipeline = dai.Pipeline()

# Create input control node to acquire capture command
xinCaptureCommand = pipeline.create(dai.node.XLinkIn)
xinCaptureCommand.setStreamName("capture")

# Create Camera node and give its properties
camRGB = pipeline.create(dai.node.ColorCamera)
camRGB.setStillSize(640, 640)
camRGB.setPreviewSize(640, 640)
camRGB.setVideoSize(640, 640)
camRGB.setInterleaved(False)
camRGB.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# Create output node for still images
outStillRGB = pipeline.create(dai.node.XLinkOut)
outStillRGB.setStreamName("rgbStill")

# Link output of xinCaptureCommand to camera input control
xinCaptureCommand.out.link(camRGB.inputControl)

# Link output of camera to input of xlinkout to send to device
camRGB.still.link(outStillRGB.input)

# Connect to device and start the pipeline
with dai.Device(pipeline) as device:
    # Create input queue to device, that receives capture command
    captureInputQueue = device.getInputQueue("capture")
    
    # Create output queue that will get RGB frame (Output from device, and input to host)
    stillQueue = device.getOutputQueue(name="rgbStill")
    
    cv2.imshow("blue", blue)
    
    # Make sure the destination path is present before starting
    dirName = "rgb_data"
    Path(dirName).mkdir(parents=True, exist_ok=True)
    
    print("Waiting for capture command...")
    
    while True:
        stillFrame = stillQueue.tryGet()
        if stillFrame is not None:
            print("Captured frame")
            print(stillFrame.getHeight(), stillFrame.getWidth(), stillFrame.getType())
            frame = stillFrame.getCvFrame()
            # Show still frame
            cv2.imshow("frame", frame)
            # Save still frame to file
            fName = f"{dirName}/{int(time.time()*1000)}.jpeg"
            cv2.imwrite(fName, frame)

        # Send capture command from host to device
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord('c'):
            ctrl = dai.CameraControl()
            ctrl.setCaptureStill(True)
            captureInputQueue.send(ctrl)