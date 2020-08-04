import torch
from torch.utils.data import DataLoader 
from LatentDataset import LatentDataset
from KittiRoiDataset import KittiDataset
import torchvision
import torch.nn as nn
from torchsummary import summary
from PIL import Image
import torchvision.transforms as T
import numpy as np
import glob
from torch.utils.tensorboard import SummaryWriter

class Resnet101(nn.Module):
    def __init__(self,c_dim,normalize=True):
        super().__init__()
        self.normalize = normalize
        self.features =torchvision.models.resnet101(pretrained=True)
        self.features.fc = nn.Sequential()
        self.fc = nn.Linear(2048,c_dim)
    def forward(self,x):
        net = self.features(x)
        out = self.fc(net)
        return out

def main():
    USE_GPU = True
    RUN_PARALLEL = True
    device_ids = [0, 1]
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        if torch.cuda.device_count() <= 1:
            RUN_PARALLEL = False
            pass
    else:
        device = torch.device('cpu')
        RUN_PARALLEL = False
    learning_rate = 1e-3
    learning_rate_decay = 0.3
    cfg = dict(device=device, learning_rate=learning_rate, learning_rate_decay=learning_rate_decay,
               last_observe=30, epochs=25, print_every=2, save_every=2, batch_size=20,
               data_locate="./data/forecasting_dataset/train/", save_path="./model_ckpt/",
               log_file="./log.txt", tensorboard_path="runs/train_visualization")

    train_dataset = LatentDataset()
    train_loader = DataLoader(dataset=train_dataset,batch_size=cfg['batch_size'], shuffle=True, num_workers=0, drop_last=True)
    
    model = Resnet101(384)
    model.to(device)
    if RUN_PARALLEL:
        model = nn.DataParallel(model,device_ids=device_ids)
    optimizer = torch.optim.Adadelta(model.parameters(), rho = 0.9)

    do_train(model,cfg,train_loader,optimizer)
    
def do_train(model,cfg,train_loader,optimizer,scheduler=None):
    device = cfg['device']
    print_every = cfg['print_every']
    writer = SummaryWriter(cfg['tensorboard_path'])
    for e in range(cfg['epochs']):
        for i,data in enumerate(train_loader):
            model.train()
            img, gt_latent_code = data
            img = img.to(device=device,dtype=torch.float)
            pred_latent_code = model(img)
            gt_latent_code = gt_latent_code.to(device,dtype=torch.float)
            criterion = nn.MSELoss(reduction = 'mean') 
            loss = criterion(pred_latent_code,gt_latent_code)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % print_every == 0:
                writer.add_scalar('training_loss',loss.item(),e)
                print('Epoch %d/5: Iteration %d, loss = %.4f' % (e+1, i, loss.item()))

    torch.save(model.state_dict(), cfg['save_path']+"Resnet101.pth")

if __name__ ==  "__main__":
   main()
   
   