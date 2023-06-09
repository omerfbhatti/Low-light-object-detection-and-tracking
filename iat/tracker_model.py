import torch
#import torch.nn as nn
#import torchvision
import torchvision.transforms as transforms
import numpy as np
#import cv2
#import sys

from iat.IAT_main import IAT

#from base_model import base_detector
from base_model_deepSORT import base_detector

class iat_detector(base_detector):
    def __init__(self, YOLO_MODEL_PATH = './weights/yolov5s_iat.pt',
                 ENHANCEMENT_MODEL_PATH = "./weights/best_Epoch_lol_v1.pth"):
        
        super().__init__(YOLO_MODEL_PATH)

        self.enhancement_model = IAT()
        self.load_network(self.enhancement_model, ENHANCEMENT_MODEL_PATH)
        self.enhancement_model = self.enhancement_model.to(self.device)
        self.enhancement_model.eval()

        self.transform = transforms.ToTensor()

    def enhance_image(self, img):
        img = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _,_,img = self.enhancement_model(img)
        
        img = self.tensor2im(img)
        return img
    
    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].detach().cpu().float().numpy()
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0 #+ 1) / 2.0 * 255.0
        #image_numpy = np.maximum(image_numpy, 0)
        #image_numpy = np.minimum(image_numpy, 255)
        return image_numpy.astype(imtype)