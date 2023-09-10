
from ia_ultralytics.engine.IPAM import IA_Model
from ia_ultralytics import YOLO
import copy
from torch import nn
import torch
if __name__=="__main__":


    model = YOLO("yolov8n.yaml")  # build a newcfg model from scratch
    ia=model.model.model[0]
    #print("m2: ",ia)

    x=torch.rand((1,3,640,640)).half()
    print(x.dtype)