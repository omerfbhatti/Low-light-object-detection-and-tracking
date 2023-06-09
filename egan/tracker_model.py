import torch
#import torch.nn as nn
#import torchvision
import torchvision.transforms as transforms
import numpy as np
#import cv2
#import sys

from egan.networks import define_G
from egan.opt import options

#from base_model import base_detector
from base_model_deepSORT import base_detector

opt = options()

#YOLO_MODEL_PATH = './yolov5s_eganjt.pt'

class egan_detector(base_detector):
    def __init__(self, YOLO_MODEL_PATH = './weights/yolov5s_eganjt.pt',
                 ENHANCEMENT_MODEL_PATH = "./weights/egat-jt_40-epoch.pth"):
        
        super().__init__(YOLO_MODEL_PATH)
        self.gpu_ids = [0]

        self.enhancement_model = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, 
                                        opt.norm, not opt.no_dropout, self.gpu_ids, skip=opt.skip, opt=opt)
        self.load_network(self.enhancement_model, ENHANCEMENT_MODEL_PATH)
        self.enhancement_model.eval()

        transform_list = [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
        self.transforms = transforms.Compose(transform_list)

    
    def enhance_image(self, img):
        img, img_gray = self.imgToTensors(img)
        with torch.no_grad():
            enhanced_img,_ = self.enhancement_model.forward(img, img_gray)
        #print("Output of enhancement: ", img.shape)
        img = np.ascontiguousarray(self.tensor2im(enhanced_img))
        #cv2.imwrite("np_img.jpg", img)
        return img
        

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
    
    