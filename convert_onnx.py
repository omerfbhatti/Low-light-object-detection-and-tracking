import torch
import io
import cv2
import numpy as np
import torch.nn as nn
from torch.utils import model_zoo
import torch.onnx

SOURCE_VIDEO = "./singapore_morning_drive.mp4"


if __name__=="__main__":
    from egan.tracker_model import egan_detector
    torch_model = egan_detector()

    # Open video stream
    cap = cv2.VideoCapture(SOURCE_VIDEO)
    success, x = cap.read()

    torch_out = torch_model(x)

    # Export the model
    torch.onnx.export(torch_model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    "enhanced_yolo.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})
