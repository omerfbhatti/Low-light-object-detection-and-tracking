import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from SCI.model import Finetunemodel

class SCI(object):
    def __init__(self, ENHANCEMENT_MODEL_PATH = "./weights/sci_difficult.pt"):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.enhancement_model = Finetunemodel(weights=ENHANCEMENT_MODEL_PATH)
        #self.load_network(self.enhancement_model, ENHANCEMENT_MODEL_PATH)
        self.enhancement_model = self.enhancement_model.to(self.device)
        self.enhancement_model.eval()

        transform_list = [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def __call__(self, img):
        img = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _,enhanced_image = self.enhancement_model(img)
        #img = np.ascontiguousarray(self.tensor2im(enhanced_image))
        img = self.tensor2im(enhanced_image)
        return img
    
    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].detach().cpu().float().numpy()
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0 #+ 1) / 2.0 * 255.0
        image_numpy = np.maximum(image_numpy, 0)
        image_numpy = np.minimum(image_numpy, 255)
        image = Image.fromarray(image_numpy.astype(imtype), 'RGB')
        return image

if __name__=="__main__":
    transform = transforms.Compose([SCI()])
    image = Image.open("egan_img.jpg")
    image = transform(image)
    image.show()