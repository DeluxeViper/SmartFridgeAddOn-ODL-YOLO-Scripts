import cv2
import depthai as dai
import time
from pathlib import Path
from PIL import Image
import numpy as np
import argparse
import sys
import json

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Provide model name or model path for inference",
                    default='yolov4_tiny_coco_416x416', type=str)
parser.add_argument("-c", "--config", help="Provide config path for inference",
                    default='json/yolov4-tiny.json', type=str)
args = parser.parse_args()

# parse config
configPath = Path(args.config)
if not configPath.exists():
    raise ValueError("Path {} does not exist!".format(configPath))

with configPath.open() as f:
    config = json.load(f)
nnConfig = config.get("nn_config", {})

# parse input shape
if "input_size" in nnConfig:
    W, H = tuple(map(int, nnConfig.get("input_size").split('x')))

# extract metadata
metadata = nnConfig.get("NN_specific_metadata", {})
classes = metadata.get("classes", {})
coordinates = metadata.get("coordinates", {})
anchors = metadata.get("anchors", {})
anchorMasks = metadata.get("anchor_masks", {})
iouThreshold = metadata.get("iou_threshold", {})
confidenceThreshold = metadata.get("confidence_threshold", {})

print(metadata)

# parse labels
nnMappings = config.get("mappings", {})
labels = nnMappings.get("labels", {})

# get model path
nnPath = args.model
if not Path(nnPath).exists():
    print("No blob found at {}. Looking into DepthAI model zoo.".format(nnPath))
    nnPath = str(blobconverter.from_zoo(args.model, shaves = 6, zoo_type = "depthai", use_cache=True))
# sync outputs
syncNN = True

blue = cv2.imread("blue.jpeg")

# Create pipeline
pipeline = dai.Pipeline()

manip = pipeline.create(dai.node.ImageManip)
manip.initialConfig.setResize(640, 640)
manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
manip.setMaxOutputFrameSize(640*640*3)

# Create input control node to acquire capture command
xinCaptureCommand = pipeline.create(dai.node.XLinkIn)
xinCaptureCommand.setStreamName("capture")

# Create Camera node and give its properties
camRGB = pipeline.create(dai.node.ColorCamera)
# camRGB.setStillSize(640, 640)
# camRGB.setPreviewSize(640, 640)
# camRGB.setVideoSize(640, 640)
# camRGB.setInterleaved(False)
camRGB.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
camRGB.setFp16(True)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
# Create output node for neural network
nnOut = pipeline.create(dai.node.XLinkOut)
nnOut.setStreamName("nn")

# Create output node for still images
outStillRGB = pipeline.create(dai.node.XLinkOut)
outStillRGB.setStreamName("rgbStill")

# Network specific settings
detectionNetwork.setConfidenceThreshold(confidenceThreshold)
detectionNetwork.setNumClasses(classes)
detectionNetwork.setCoordinateSize(coordinates)
detectionNetwork.setAnchors(anchors)
detectionNetwork.setAnchorMasks(anchorMasks)
detectionNetwork.setIouThreshold(iouThreshold)
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

# Linking
# manip.out.link(detectionNetwork.input)
# camRGB.still.link(detectionNetwork.input)

# Link output of xinCaptureCommand to camera input control
xinCaptureCommand.out.link(camRGB.inputControl)

# Link output of camera to input of xlinkout to send to device
# camRGB.still.link(outStillRGB.input)

# camRGB.still.link(detectionNetwork.input)
camRGB.still.link(manip.inputImage)
manip.out.link(detectionNetwork.input)
# Link output of detectionNetwork to camera input
detectionNetwork.passthrough.link(outStillRGB.input)

# Link output of detectionNetwork to NN input
detectionNetwork.out.link(nnOut.input)

# Connect to device and start the pipeline
with dai.Device(pipeline) as device:
    # Set debugging level
    # device.setLogLevel(dai.LogLevel.DEBUG)
    # device.setLogOutputLevel(dai.LogLevel.DEBUG)

    print(f"dnet: {detectionNetwork.getCoordinateSize()}")
    # Create input queue to device, that receives capture command
    captureInputQueue = device.getInputQueue("capture")
    
    # Create output queue that will get RGB frame (Output from device, and input to host)
    stillQueue = device.getOutputQueue(name="rgbStill")

    # Create output queue that will get detections (Output from NN ?)
    detectionQueue = device.getOutputQueue(name="nn")
    
    detections = []
    counter = 0
    color2 = (255, 255, 255)
    detectionDirName = "detection_data"

    object_count = {'Orange': 0, 'Apple': 0, 'Strawberry': 0, 'Tomato': 0}
    
    cv2.imshow("blue", blue)
    
    # Make sure the destination path is present before starting
    rgbDirName = "rgb_data"
    Path(rgbDirName).mkdir(parents=True, exist_ok=True)
    #Path(detectionDirName).mkdir(parents=True, exist_ok=True)

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        print("In frame norm")
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame, detections):
        # print(frame)
        print("Displaying frame:")
        color = (255, 0, 0)
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        # Show the frame
        cv2.imshow(name, frame)

    print("Waiting for capture command...")    
    while True:
        inDet = detectionQueue.tryGet()
        stillFrame = stillQueue.tryGet()

        if stillFrame is not None:
            print("Captured frame")
            print(stillFrame.getHeight(), stillFrame.getWidth(), stillFrame.getType())
            frame = stillFrame.getCvFrame()

            displayFrame("rgb", frame, detections)
            # # Show still frame
            # cv2.imshow("frame", frame)
            # # Save still frame to file
            # fName = f"{rgbDirName}/{int(time.time()*1000)}.jpeg"
            # cv2.imwrite(fName, frame)
    
        if inDet is not None:
            print("Detected objects:")
            detections = inDet.detections
            counter += 1
            for detection in detections:
               object_count[labels[detection.label]] += 1
            print(object_count)

            # Write to file
            # fName = f"{dirName}/{int(time.time() * 1000)}.txt"
            fName = "detected_objects.txt"
            with open(fName, "wb") as f:
                for k, v in object_count.items():
                    b = bytes(f"{k} {v}\n", 'utf-8')
                    f.write(b)
                print('detection data saved to', fName)

        # Send capture command from host to device
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == ord('c'):
            for detection in detections:
               object_count[labels[detection.label]] = 0
            ctrl = dai.CameraControl()
            ctrl.setCaptureStill(True)
            captureInputQueue.send(ctrl)