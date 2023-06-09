import torch
import torch.nn as nn
#import torchvision
#import torchvision.transforms as transforms
import numpy as np
import cv2
import sys

# ByteTracking Utilities
sys.path.append(f"./ByteTrack/")
from dataclasses import dataclass
from yolox.tracker.byte_tracker import BYTETracker
from onemetric.cv.utils.iou import box_iou_batch
from byte_tracker_utils import *


class base_detector(nn.Module):
    def __init__(self, YOLO_MODEL_PATH="./weights/yolov5s_lowlight.pt"):
        super().__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.enhancement_model = None

        self.detector_model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH).to(self.device)
        self.classes = self.detector_model.names

        # Object Tracker
        self.object_tracker = BYTETracker(BYTETrackerArgs())
        #self.annotator = BaseAnnotator(colors=COLORS, thickness=2)


    def forward(self, img, enhance=True, return_image=False):
        # STEP 1: ENHANCE IMAGE
        if enhance:
            img = self.enhance_image(img)

        # STEP 2: PERFORM DETECTION (YOLOv5)
        with torch.no_grad():
            results = self.detector_model(img)

        # STEP 3: TRACK OBJECTS (byteTRACK)
        detections = Detection.from_results(pred=results.pred[0].cpu().numpy(), 
                                            names=self.classes)
        #print("detections: ", detections)
        if return_image:
            result = {'detections':[], 'enhanced_image':img}
        else:
            result = {'detections':[], 'enhanced_image':None}

        if len(detections) != 0:
            tracks = self.object_tracker.update(output_results=detections2boxes(detections=detections),
                                            img_info=img.shape, img_size=img.shape)
            #print("tracks: ", tracks)
            if len(tracks) != 0:
                detections = match_detections_with_tracks(detections=detections, tracks=tracks)
                result['detections'] = detections
            else:
                result['detections'] = None
        else:
            result['detections'] = None
        
        return result
    
    def enhance_image(self, img):
        return img
    
    def class_to_label(self, x):
        return self.classes[int(x)]
    
    def load_network(self, network, save_path):
        network.load_state_dict(torch.load(save_path))