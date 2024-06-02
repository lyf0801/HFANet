import numpy as np
import torch
import torch.utils.data as Data
import os
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from ORSI_SOD_dataset import ORSI_SOD_dataset
from tqdm import tqdm


from src.HFANet_R import net as Net_R
from src.HFANet_V import net as Net_V

from evaluator import Eval_thread
from PIL import Image


os.environ['CUDA_VISIBLE_DEVICES'] = '0'





def unload(x):
    y = x.squeeze().cpu().data.numpy()
    return y
def convert2img(x):
    return Image.fromarray(x*255).convert('L')
def min_max_normalization(x):
    x_normed = (x - np.min(x)) / (np.max(x)-np.min(x))
    return x_normed
def save_smap(smap, path, negative_threshold=0.25):
    # smap: [1, H, W]
    if torch.max(smap) <= negative_threshold:
        smap[smap<negative_threshold] = 0
        smap = convert2img(unload(smap))
    else:
        smap = convert2img(min_max_normalization(unload(smap)))
    smap.save(path)



def getsmaps(dataset_name, model_Type):

    dataset_root = "/data/iopen/lyf/SaliencyOD_in_RSIs/" + dataset_name + " dataset/"
    test_set = ORSI_SOD_dataset(root = dataset_root, mode = "test", aug = False)
    test_loader = DataLoader(test_set, batch_size = 1, num_workers = 1)
    

    if model_Type == "HFANet_R":
        net = Net_R().cuda().eval() 
        if dataset_name == "ORSSD":
            net.load_state_dict(torch.load("./data/HFANet_R_weights/ORSSD_weights.pth", map_location='cuda:0'))
        elif dataset_name == "EORSSD":
            net.load_state_dict(torch.load("./data/HFANet_R_weights/EORSSD_weights.pth", map_location='cuda:0'))
        elif dataset_name == "ORS_4199":
            net.load_state_dict(torch.load("./data/HFANet_R_weights/ORS_4199_weights.pth", map_location='cuda:0'))
    
    elif model_Type == "HFANet_V":
        net = Net_V().cuda().eval()  
        if dataset_name == "ORSSD":
            net.load_state_dict(torch.load("./data/HFANet_V_weights/ORSSD_weights.pth", map_location='cuda:0'))
        elif dataset_name == "EORSSD":
            net.load_state_dict(torch.load("./data/HFANet_V_weights/EORSSD_weights.pth", map_location='cuda:0'))
        elif dataset_name == "ORS_4199":
            net.load_state_dict(torch.load("./data/HFANet_V_weights/ORS_4199_weights.pth", map_location='cuda:0'))    
    
    net.eval()


    for image, label, edge, name in tqdm(test_loader):  
    
        with torch.no_grad():
            image, label = image.cuda(), label.cuda()
            
            smap = net(image)  
            
            ##create file dirs
            dirs = "./data/output/predict_smaps" +  "_" + model_Type + "_" + dataset_name
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            path = os.path.join(dirs, name[0] + "_" + model_Type + '.png')  
            save_smap(smap, path)

if __name__ == "__main__":
    
    dataset = ["ORSSD", "EORSSD", "ORS_4199"]
    model_Type = "HFANet_R"
    #model_Type = "HFANet_V"
    for datseti in dataset:
        getsmaps(datseti, model_Type)
