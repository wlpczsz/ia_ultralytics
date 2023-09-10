import os
os.environ["original_yolo"]="False"
from ia_ultralytics import YOLO
import ipdb
if __name__ == '__main__':
    # Load a modelmodel
    model = YOLO("yolov8n.yaml")  # build a newcfg model from scratch
    #model = YOLO(r"runs\detect\IA-yolov8n\weights\best.pt")  # load a pretrained model (recommended for training)
    #ipdb.set_trace()
    # Use the model
    model.train(data="coco128.yaml",epochs=100)  # train the model
    #metrics = model.val(data="coco128.yaml",imgsz=640)  # evaluate model performance on the validation set
    #results = model("https://ia_ultralytics.com/images/bus.jpg")  # predict on an image
    path = model.export(format="onnx")  # export the model to ONNX format