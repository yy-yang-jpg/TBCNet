import os
import torch
from torch.utils.data import DataLoader
from model.dataloader import MVA_Dataset, Normalize0, ShortSideScale
from model.dataloader import build_config

import argparse
from torchvision.transforms import (
    CenterCrop,
    Compose,
)

from model.MVA import TBCNet, TBCNet_test

def evaluate(label, pred):
    num_all=len(label)
    num_right=sum(a == b for a, b in zip(label, pred))
    acc=num_right/num_all
    return acc

def valid(model, device, loader):
    model.eval()
    y_pred_vector = []
    labels_vector = []
    for step, (input, y) in enumerate(loader):
                
        input = input.transpose(0, 1).to(device)
        y=(y-1).to(device)
        with torch.no_grad():
            y_p = model(input) 
            y_pred = torch.argmax(y_p, dim=1)
            y_pred = y_pred.detach() 
        
        y_pred_vector.extend(y_pred)  
        labels_vector.extend(y)             
            
    print('---------Test---------')
    print('Results:')
    acc = evaluate(labels_vector, y_pred_vector)
    print(f'ACC = {acc*100}%')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--checkpoint', default=None, 
                        type=str, required=False, help='Path to the pre-trained model.')
    parser.add_argument("--testtype", dest='testtype', default="CS", choices=["CS", 'CV','CSet']
                        , required=False, help='Set test type from CS or CV')
    parser.add_argument('--dataset', type=str, default='ntu-60', required=False, help='Dataset to use.', 
                        choices=["ntu-120", 'ntu-60',"pkummd", "n-ucla"])
    parser.add_argument("--device", dest='device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                        , required=False, help='Set CUDA_VISIBLE_DEVICES environment variable, optional')
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument("--num_workers", default=8)
    args = parser.parse_args()
    batch_size,num_workers,= args.batch_size,args.num_workers
    device=args.device
    high_feature_dim, low_feature_dim= 128,256
    if args.testtype=='CV':
        view_num=2
    else:
        view_num=3
  
    transform_test = Compose(
        [
            Normalize0([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
            ShortSideScale(size=256),
            CenterCrop(224)
        ]
    )
    cfg = build_config(args.dataset,args.testtype)
    data_gen = MVA_Dataset(cfg, 'test', transform=transform_test) 
    
    data_loader = DataLoader(data_gen, batch_size=args.batch_size, shuffle = True,
                                num_workers=args.num_workers,pin_memory=True)
    
    model = TBCNet_test(view_num, low_feature_dim, high_feature_dim,cfg.num_actions,device,"test")
    checkpoint0 = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint0)
    model = model.to(args.device)
    print("Dataset:{}".format(args.dataset))
    print("Loading models...")
    valid(model, args.device, data_loader, view_num)