<!-- @format -->

# Yolo detection with on-device decoding

Run model using this command:

python3 yolo_odl_detection.py -m yolov7appleorangestrawberrytomato_openvino_2021.4_6shave.blob --config yolov7appleorangestrawberrytomato.json

Run pipeline graph using this command:

pipeline_graph run "python3 still_image_yolo_detection.py -m yolov7appleorangestrawberrytomato_openvino_2021.4_6shave.blob --config yolov7appleorangestrawberrytomato.json"

## Usage

1. Install requirements
   ```python
   python3 -m pip install -r requirements.txt
   ```
2. Run the script

   ```
   python3 main.py --config <config_json>
   ```

   or

   ```
   python3 main_api.py -m <model_name> --config <config_json>
   ```

   where:

   - `<model_name>` is the **name of the model** from DepthAI model zoo (https://zoo.luxonis.com) or **relative path to the blob** file. Please check our model zoo to see which pre-trained models are available.
   - `<config_json>` is the **relative path** to the JSON with metadata (input shape, anchors, labels, ...) of the Yolo model.
