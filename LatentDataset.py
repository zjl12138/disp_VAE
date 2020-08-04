import torch
import torch.utils.data
import torchvision.transforms as T
import numpy as np
from PIL import Image
import glob
import os


def write_latent_code(array, save_path):
    with open(save_path, 'w') as f:
        np.savetxt(f, array, delimiter=' ', fmt='%f %f')


def read_latent_code(file_path):
    #with open(file_path,encoding='gbk') as f:
    #print(file_path)
    latent = np.loadtxt(file_path)
    z_mean = latent[:, 0]
    z_log_var = latent[:, 1]
    return torch.tensor(z_mean), torch.tensor(z_log_var)


def read_image(file_path):
    loader = T.Compose([
    T.ToTensor()])  
    img = Image.open(file_path)
    #return loader(img)
    return img
    
class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, transforms=None):
        super().__init__()
        self.root_dir = "./datasets/roi_result/left"
        self.latent_root_dir = "./datasets/latent_result"

        self.latent_file_names = glob.glob(self.latent_root_dir+"/*/*.txt")

        self.file_names = glob.glob(self.root_dir + "/*/*.png")

        if transforms is not None:
            self.transform = transforms
        else:

            self.transform = T.Compose([
                
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                # T.Normalize(mean=[0.485, 0.456, 0.406],
                #             std=[0.229, 0.224, 0.225]) #TODO
                ])
        
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        latent_path = self.latent_file_names[index]
        #(filepath, tempfilename) = os.path.split(img_path)
        #img_meta = dict(img_id=os.path.split(filepath)[1], car_id=os.path.splitext(tempfilename)[0])
        img = read_image(img_path)
        
        img = self.transform(img)
 
        latent_path = self.latent_file_names[index]  
        z_mean, z_log_var = read_latent_code(latent_path)
        epsilon = torch.randn(z_mean.size()[0])
        latent_code = z_mean + torch.exp(z_log_var) * epsilon  # B * latent_num
        return img, latent_code
