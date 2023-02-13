import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
import keras
import sys

from mbllen.network import build_mbllen

from base_model import base_detector

ENHANCEMENT_MODEL_PATH = "./weights/Syn_img_lowlight_withnoise.h5"

lowpercent = 5
highpercent = 95
maxrange = 8/10.
hsvgamma = 8/10.


class mbllen_detector(base_detector):
    def __init__(self, YOLO_MODEL_PATH = './weights/yolov5s_mbllen.pt'):
        super().__init__(YOLO_MODEL_PATH)

        self.enhancement_model = build_mbllen((None, None, 3))
        self.enhancement_model.load_weights(ENHANCEMENT_MODEL_PATH)
        opt = keras.optimizers.Adam(lr=2 * 1e-04, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.enhancement_model.compile(loss='mse', optimizer=opt)
        #self.enhancement_model = self.enhancement_model.to(self.device)
        #self.enhancement_model.eval()

    def enhance_image(self, img):
        img = (np.asarray(img)/255.0)
        img = img[np.newaxis, :]

        with torch.no_grad():
            img = self.enhancement_model.predict(img)
        
        img = self.postprocess_image(img)
        
        return img
    
    def postprocess_image(self, img):
        fake_B = img[0, :, :, :3]
        #fake_B_o = fake_B

        gray_fake_B = fake_B[:, :, 0] * 0.299 + fake_B[:, :, 1] * 0.587 + fake_B[:, :, 1] * 0.114
        percent_max = sum(sum(gray_fake_B >= maxrange))/sum(sum(gray_fake_B <= 1.0))
        # print(percent_max)
        max_value = np.percentile(gray_fake_B[:], highpercent)
        if percent_max < (100-highpercent)/100.:
            scale = maxrange / max_value
            fake_B = fake_B * scale
            fake_B = np.minimum(fake_B, 1.0)

        gray_fake_B = fake_B[:,:,0]*0.299 + fake_B[:,:,1]*0.587 + fake_B[:,:,1]*0.114
        sub_value = np.percentile(gray_fake_B[:], lowpercent)
        fake_B = (fake_B - sub_value)*(1./(1-sub_value))

        imgHSV = cv2.cvtColor(fake_B, cv2.COLOR_RGB2HSV)
        H, S, V = cv2.split(imgHSV)
        S = np.power(S, hsvgamma)
        imgHSV = cv2.merge([H, S, V])
        fake_B = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2RGB)
        fake_B = np.minimum(fake_B, 1.0)

        fake_B = np.minimum(fake_B, 1.0)
        fake_B = np.maximum(fake_B, 0.0)
        fake_B = (fake_B * 255).astype(np.uint8)

        return fake_B