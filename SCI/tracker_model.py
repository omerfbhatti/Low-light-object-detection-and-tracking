import torch
import torchvision.transforms as transforms
import numpy as np

from SCI.model import Finetunemodel

from base_model import base_detector
#from base_model_deepSORT import base_detector

class sci_detector(base_detector):
    def __init__(self, YOLO_MODEL_PATH = './weights/yolov5s_sci.pt',
                 ENHANCEMENT_MODEL_PATH = "./weights/sci_difficult.pt"):
        
        super().__init__(YOLO_MODEL_PATH)

        self.enhancement_model = Finetunemodel(weights=ENHANCEMENT_MODEL_PATH)
        #self.load_network(self.enhancement_model, ENHANCEMENT_MODEL_PATH)
        self.enhancement_model = self.enhancement_model.to(self.device)
        self.enhancement_model.eval()

        transform_list = [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)


    def enhance_image(self, img):
        img = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _,enhanced_image = self.enhancement_model(img)
        img = np.ascontiguousarray(self.tensor2im(enhanced_image))
        return img

    def img2tensor(self, img):
        img = (np.asarray(img)/255.0)
        img = torch.from_numpy(img).float()
        img = img.permute(2,0,1)
        img = img.to(self.device).unsqueeze(0)
        return img
    
    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].detach().cpu().float().numpy()
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0 #+ 1) / 2.0 * 255.0
        image_numpy = np.maximum(image_numpy, 0)
        image_numpy = np.minimum(image_numpy, 255)
        return image_numpy.astype(imtype)