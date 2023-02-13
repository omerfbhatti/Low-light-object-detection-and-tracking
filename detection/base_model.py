import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import sys

# ByteTracking Utilities
sys.path.append(f"/home/dell/Desktop/joint_inference/ByteTrack/")
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
    
    def plot_boxes(self, results, frame, height, width, confidence=0.1):
        labels, cord = results
        detections = []
        n = len(labels)
        x_shape, y_shape = width, height

        for i in range(n):
            row = cord[i]

            if row[4] >= confidence:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                obj_class = self.class_to_label(labels[i])
                if obj_class in ['Car','Bus','Person', 'motorcycle']:
                    x_center = x1+(x2-x1)
                    y_center = y1+((y2-y1)/2)
                    tlwh = np.asarray([x1, y1, int(x2-x1), int(y2-y1)], dtype=np.float32)
                    confidence = float(row[4].item())
                    #feature = 'car'
                    detections.append(([x1,y1,int(x2-x1),int(y2-y1)], row[4].item(), obj_class))
                    frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),(0,0,255), 2)
                    #frame = cv2.putText(frame, "ID: "+str(track_id)+" "+track.det_class, (int(bbox[0]), int(bbox[1]-10)),
                    #    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        return frame, detections
    
    def load_network(self, network, save_path):
        network.load_state_dict(torch.load(save_path))