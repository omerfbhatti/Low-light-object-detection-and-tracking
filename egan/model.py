import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2

from egan.networks import define_G
from egan.opt import options

# ByteTracking Utilities
sys.path.append(f"/home/dell/Desktop/joint_inference/ByteTrack/")
from dataclasses import dataclass
from yolox.tracker.byte_tracker import BYTETracker
from onemetric.cv.utils.iou import box_iou_batch
from byte_tracker_utils import *


opt = options()
ENHANCEMENT_MODEL_PATH = "./egat-jt_40-epoch.pth"
YOLO_MODEL_PATH = './yolov5s_eganjt.pt'

class egan_detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.gpu_ids = [0]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.enhancement_model = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, 
                                        opt.norm, not opt.no_dropout, self.gpu_ids, skip=opt.skip, opt=opt)
        self.load_network(self.enhancement_model, ENHANCEMENT_MODEL_PATH)
        self.enhancement_model.eval()

        transform_list = [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
        self.transforms = transforms.Compose(transform_list)

        self.detector_model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH).to(self.device)
        self.classes = self.detector_model.names

        # Object Tracker
        self.object_tracker = BYTETracker(BYTETrackerArgs())
        #self.annotator = BaseAnnotator(colors=COLORS, thickness=2)


    def forward(self, img, enhance=True):
        if enhance:
            img, img_gray = self.imgToTensors(img)
            enhanced_img,_ = self.enhancement_model.forward(img, img_gray)
            #print("Output of enhancement: ", img.shape)
            img = np.ascontiguousarray(self.tensor2im(enhanced_img))
            #cv2.imwrite("np_img.jpg", img)

        results = self.detector_model(img)

        split_results = results.xyxyn[0][:,-1], results.xyxyn[0][:,:-1]
        return results, split_results
    
    #def set_input(self, img):
    #    # load images into class variables for processing 
    #    self.img.resize_(img.size()).copy_(img)
    #    self.img_gray.resize_(img_gray.size()).copy_(img_gray)
        

    def imgToTensors(self, img):
        img = self.transforms(img).to(self.device)
                
        # attention map
        r,g,b = img[0]+1, img[1]+1, img[2]+1
        img_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
        #torchvision.utils.save_image(img_gray, "img_attention.jpg")
        #torchvision.utils.save_image(img, "img.jpg")

        img_gray = torch.unsqueeze(img_gray, 0)
        img_gray = torch.unsqueeze(img_gray, 0)
        img = torch.unsqueeze(img, 0)

        #print("Calculated attention map of image")
        #print("Shape of img: ", img.shape)
        #print("Shape of img_gray: ", img_gray.shape)

        return img, img_gray
    
    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].detach().cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        image_numpy = np.maximum(image_numpy, 0)
        image_numpy = np.minimum(image_numpy, 255)
        return image_numpy.astype(imtype)
    
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