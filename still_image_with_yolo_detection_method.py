import cv2
import depthai as dai
from pathlib import Path
import numpy as np
import json
import base64

def takePhotoAndDetectObjects(model_path, config_path):
    configPath = Path(config_path)
    if not configPath.exists():
        raise ValueError("Path {} does not exist!".format(configPath))

    with configPath.open() as f:
        config = json.load(f)
    nnConfig = config.get("nn_config", {})

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
    if not Path(model_path).exists():
        print("No blob found at {}. Looking into DepthAI model zoo.".format(model_path))
        model_path = str(blobconverter.from_zoo(model_path, shaves = 6, zoo_type = "depthai", use_cache=True))
    # sync outputs
    syncNN = True

    # Create pipeline
    pipeline = dai.Pipeline()

    # Create ImageManip node
    manip = pipeline.create(dai.node.ImageManip)
    manip.initialConfig.setResize(640, 640)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    manip.setMaxOutputFrameSize(640*640*3)

    # Create input control node to acquire capture command
    xinCaptureCommand = pipeline.create(dai.node.XLinkIn)
    xinCaptureCommand.setStreamName("capture")

    # Create Camera node and give its properties
    camRGB = pipeline.create(dai.node.ColorCamera)
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
    detectionNetwork.setBlobPath(model_path)
    detectionNetwork.setNumInferenceThreads(2)
    detectionNetwork.input.setBlocking(False)

    # Linking

    # Link output of xinCaptureCommand to camera input control
    xinCaptureCommand.out.link(camRGB.inputControl)

    camRGB.still.link(manip.inputImage)
    manip.out.link(detectionNetwork.input)

    # Link output of detectionNetwork to camera input
    detectionNetwork.passthrough.link(outStillRGB.input)

    # Link output of detectionNetwork to NN input
    detectionNetwork.out.link(nnOut.input)


    # Connect to device and start the pipeline
    with dai.Device(pipeline) as device:
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
        
        # cv2.imshow("blue", blue)
        
        # Make sure the destination path is present before starting
        rgbDirName = "rgb_data"
        Path(rgbDirName).mkdir(parents=True, exist_ok=True)

        # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
        def frameNorm(frame, bbox):
            normVals = np.full(len(bbox), frame.shape[0])
            normVals[::2] = frame.shape[1]
            return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

        def displayFrameWithBB(name, frame, detections):
            color = (255, 0, 0)
            for detection in detections:
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.putText(frame, labels[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            # Show the frame
            cv2.imshow(name, frame)

        # print("Waiting for capture command...")
        capturedStillFrame = False
        detectedObjects = False
        while not capturedStillFrame and not detectedObjects:
            ctrl = dai.CameraControl()
            ctrl.setCaptureStill(True)
            captureInputQueue.send(ctrl)
            inDet = detectionQueue.tryGet()
            stillFrame = stillQueue.tryGet()

            if stillFrame is not None:
                print("Captured frame: ", stillFrame.getHeight(), stillFrame.getWidth(), stillFrame.getType())
                frame = stillFrame.getCvFrame()

                # Display frame with bounding boxes (broken)
                # displayFrameWithBB("rgb", frame, detections)

                # # Show still frame
                # cv2.imshow("frame", frame)

                # # Save still frame to file
                fImageName = f"still_image.jpeg"
                cv2.imwrite(fImageName, frame)

                # Save still frame text to file
                fImageTxtName = f"still_image.txt"
                retval, buffer = cv2.imencode('.jpg', frame)
                jpg_as_text = base64.b64encode(buffer)
                with open(fImageTxtName, "wb") as f:
                    f.write(jpg_as_text)
                print('still image frame saved to', fImageName)
                capturedStillFrame = True
                

            if inDet is not None:
                print("Detected objects:")
                detections = inDet.detections
                counter += 1
                for detection in detections:
                    object_count[labels[detection.label]] += 1
                    print(f"Object: {labels[detection.label]}, Confidence: {detection.confidence}")
                print(object_count)

                # Write to file
                # fName = f"{dirName}/{int(time.time() * 1000)}.txt"
                fName = "detected_objects.txt"
                with open(fName, "wb") as f:
                    for k, v in object_count.items():
                        b = bytes(f"{k} {v}\n", 'utf-8')
                        f.write(b)
                    print('detection data saved to', fName)

                detectedObjects = True