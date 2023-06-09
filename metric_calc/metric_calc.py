import torch
import numpy as np
from IQA_pytorch import SSIM, MS_SSIM



# calculate PSNR
class PSNR(nn.Module):
    def __init__(self, max_val=0):
        super().__init__()

        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()

        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20 * torch.log(max_val) / base10)

    def __call__(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)

        if mse == 0:
            return 0

        return 10 * torch.log10((1.0 / mse))
    

if __name__=="__main__":
    ssim = SSIM()
    psnr = PSNR()
    ssim_list = []
    psnr_list = []

    for i, imgs in tqdm(enumerate(val_loader)):
        #print(i)
        low_img, high_img, name = imgs[0].cuda(), imgs[1].cuda(), str(imgs[2][0])
        print(name)
        #print(low_img.shape)

        ssim_value = ssim(enhanced_img, high_img, as_loss=False).item()
        psnr_value = psnr(enhanced_img, high_img).item()

        ssim_list.append(ssim_value)
        psnr_list.append(psnr_value)


    SSIM_mean = np.mean(ssim_list)
    PSNR_mean = np.mean(psnr_list)
    print('The SSIM Value is:', SSIM_mean)
    print('The PSNR Value is:', PSNR_mean)
