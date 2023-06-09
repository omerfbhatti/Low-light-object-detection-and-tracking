import torch
import torch.nn as nn
#import torchvision
#import torchvision.transforms as transforms
import numpy as np
import cv2
import os

# DeepSORT Tracker
from deep_sort_realtime.deepsort_tracker import DeepSort

from byte_tracker_utils import Detection, Rect

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class base_detector(nn.Module):
    def __init__(self, YOLO_MODEL_PATH="./weights/yolov5s_lowlight.pt"):
        super().__init__()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.enhancement_model = None

        self.detector_model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH).to(self.device)
        self.classes = self.detector_model.names

        # Object Tracker
        self.object_tracker = DeepSort(max_age=25,
                                        n_init=3,
                                        nms_max_overlap=1.0,
                                        max_cosine_distance=0.25,
                                        nn_budget=None,
                                        override_track_class=None,
                                        embedder="mobilenet",
                                        half=True,
                                        bgr=True,
                                        embedder_gpu=True,
                                        embedder_model_name=None,
                                        embedder_wts=None,
                                        polygon=False,
                                        today=None)


    def forward(self, img, enhance=True, return_image=False):
        # STEP 1: ENHANCE IMAGE
        if enhance:
            img = self.enhance_image(img)

        # STEP 2: PERFORM DETECTION (YOLOv5)
        with torch.no_grad():
            results = self.detector_model(img)

        # STEP 3: TRACK OBJECTS (byteTRACK)
        split_results = results.xyxyn[0][:,-1], results.xyxyn[0][:,:-1]
        img, detections = self.plot_boxes(split_results, img, height=img.shape[0], width=img.shape[1], confidence=0.4)
        tracks = self.object_tracker.update_tracks(detections, frame=img)
        detections=[]
        #print("classes: ", self.classes)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            
            bbox = ltrb
            obj_class = track.det_class
            confidence = track.det_conf
            detections.append(Detection(
                                rect=Rect(
                                    x=float(bbox[0]),
                                    y=float(bbox[1]),
                                    width=float(bbox[2] - bbox[0]),
                                    height=float(bbox[3] - bbox[1])
                                ),
                                class_id=list(self.classes.values()).index(obj_class),
                                class_name=obj_class,
                                confidence=float(confidence) if confidence is not None else 0.5,
                                tracker_id=track_id
                            ))

        #print("detections: ", detections)
        if return_image:
            result = {'detections':detections, 'enhanced_image':img}
        else:
            result = {'detections':detections, 'enhanced_image':None}

        
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

            if row[4]: #>= confidence:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                obj_class = self.class_to_label(labels[i])
                #if obj_class in ['Car','Bus','Person', 'motorcycle']:
                x_center = x1+(x2-x1)
                y_center = y1+((y2-y1)/2)
                tlwh = np.asarray([x1, y1, int(x2-x1), int(y2-y1)], dtype=np.float32)
                confidence = float(row[4].item())
                detections.append(([x1,y1,int(x2-x1),int(y2-y1)], row[4].item(), obj_class))
            
        if len(detections)==0:
            detections.append(([0,0,0,0], 0.01, 'car'))

        return frame, detections
    
    def load_network(self, network, save_path):
        network.load_state_dict(torch.load(save_path))