from datetime import datetime
import os
import torch
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.init import kaiming_uniform_
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
)
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from .MVA import TBCNet
from .Loss import CLoss
from .dataloader import (MVA_Dataset, RandomShortSideScale, 
                                   ShortSideScale,
                                   Normalize0)

def evaluate(label, pred):
    num_all=len(label)
    num_right=sum(a == b for a, b in zip(label, pred))
    acc=num_right/num_all
    return acc
       
def train_model(cfg, save_dir, args, writer):
    batch_size,num_workers,num_epochs= args.batch_size,args.num_workers,args.num_epochs
    temperature_f,device,learning_rate=args.temperature_f,args.device,args.learning_rate
    high_feature_dim, low_feature_dim= 128, 256
    loss_name=args.loss_name
    if args.traintype=='CS'or args.traintype == 'CSet':
        view_num=3
        view_num_t=3
    elif args.traintype=='CV':
        view_num=2
        view_num_t=1
       
    transform_train = Compose(
        [
            Normalize0([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
            RandomShortSideScale(
                min_size=224,
                max_size=256,
            ),
            RandomCrop(224),
            RandomHorizontalFlip(p=0.5)
            
        ]
    )
    transform_test = Compose(
        [
            Normalize0([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
            ShortSideScale(
                size=256
            ),
            CenterCrop(224)
        ]
    )

    train_data = MVA_Dataset(cfg, 'train', transform=transform_train)
    test_data = MVA_Dataset(cfg, 'test', transform=transform_test) 
    
    train_dataloader = DataLoader(train_data, batch_size=batch_size, #collate_fn=collate_fn,
                                num_workers=num_workers,pin_memory=True, shuffle = True, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=True)
    
    print("Run ID : " + args.dataset)
    print("batch_size: " + str(batch_size))
    print("lr: " + str(learning_rate))
    print("Number of training samples : " + str(len(train_data)))
    print("Number of testing samples : " + str(len(test_data)))
    print("Steps per epoch: " + str(len(train_data) / batch_size))
    
    torch.autograd.set_detect_anomaly(True)
    model = TBCNet(view_num, low_feature_dim, high_feature_dim,cfg.num_actions,device,'FC')
    if isinstance(model, nn.Linear):
        kaiming_uniform_(model.weight.data)

    if args.checkpoint!=None:
        model_path = args.checkpoint   
        state0 = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state0)
    
    model.to(device)
    
    if args.optimizer == 'ADAM':
        print("Using ADAM optimizer")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'ADAMW':
        print("Using ADAMW optimizer")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        print("Using SGD optimizer")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    scheduler_c=0
    scaler = GradScaler(enabled=args.mixed_precision)
    loss0 = CLoss(batch_size, device,cfg.num_actions).to(device)
    
    max_fmap_score, fmap_score,min_loss = -1, -1,100
    min_loss0=100
    for epoch in range(num_epochs):
        model,train_loss = train_epoch(epoch, train_dataloader, model, optimizer, loss0, 
                            view_num,writer, device,scheduler_c,loss_name,scaler)
        print("Time: " + str(datetime.today().strftime('%m-%d_%H:%M')))
        if train_loss < min_loss0:
                min_loss= train_loss   
                save_file_path = os.path.join(save_dir, f'{model.name}_{loss_name}_train_{min_loss:.6f}.pth')
                torch.save(model.state_dict(), save_file_path) 
        if (epoch+1) % args.validation_interval == 0:
            score1,test_loss = val_epoch(epoch, test_dataloader, model,loss0, 
                                        view_num,writer, device,loss_name) 
            if max_fmap_score < fmap_score:
                max_fmap_score = fmap_score
                save_file_path = os.path.join(save_dir, f'{model.name}_{loss_name}_max_{fmap_score*100:.4f}_{test_loss:.6f}.pth')
                torch.save(model.state_dict(), save_file_path)
            
        
def train_epoch(epoch,data_loader,model,optimizer, loss0, view_num, writer, device,lr_scheduler,loss_name,scaler):
    tot_loss,act_losses = [],[] 
    y_pred_vector,labels_vector = [],[] 
    
    model.train()
    for i, (input, y) in enumerate(data_loader): 

        y=(y-1).to(device)
        input = input.transpose(0, 1).to(device)
        
        optimizer.zero_grad()
        with autocast():
            if loss_name=='c':
                y_p = model(input)
                v_loss=0
            else:
                y_p, v_p,com, vs = model(input)
                v_loss=0
                for v in range(view_num): 
                    y_v = torch.argmax(v_p[v], dim=1)
                    l0 = loss0.criterion(v_p[v], y)
                    if loss_name=='cvl':
                        v_loss += l0 + 0.5*loss0.view_Loss(com,vs[v], y, y_v,v_p[v])
                    elif  loss_name=='cv':
                        v_loss += l0 + 0.5*loss0.view_Loss(com,vs[v], y, y_v)
                del  v_p, com, vs,l0      

            com_loss = loss0.criterion(y_p, y) 
            loss = com_loss + 0.5*v_loss/view_num
            act_losses.append(com_loss.item())
            tot_loss.append(loss.item()) 
            
        if (i+1) % 1 == 0: 
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        y_p_l = torch.argmax(y_p, dim=1)
        
        y_pred_vector.extend(y_p_l)  
        labels_vector.extend(y) 
        
        del v_loss, loss, com_loss
        del y, input, i , y_p_l,y_p
        
    acc = evaluate(labels_vector, y_pred_vector)
    print(f'Training Epoch: {epoch+1:d}, Action Accuracy: {acc*100:.4f}% ,Training Loss: {np.mean(tot_loss):.4f}', 
          flush=True)
    
    return model,np.mean(tot_loss)


def val_epoch(epoch,data_loader,model,loss0, view_num_t, writer, device,loss_name):
    
    print('validation at epoch {}'.format(epoch+1))
    model.eval()
    tot_loss= []
    y_pred_vector,labels_vector = [],[] 
    for i, (input, label) in enumerate(data_loader):
        with torch.no_grad():
            input = input.transpose(0, 1).to(device)
            label=(label-1).to(device)
            if loss_name=='c':
                y_p = model(input)
                v_loss=0
            else:
                y_p, v_p,com, vs = model(input)
                v_loss=0
                for v in range(view_num_t): 
                    y_v = torch.argmax(v_p[v], dim=1)
                    l0 = loss0.criterion(v_p[v], label)
                    if loss_name=='cvl':
                        v_loss += l0 + 0.5*loss0.view_Loss(com,vs[v], label, y_v,v_p[v])
                    elif  loss_name=='cv':
                        v_loss += l0 + 0.5*loss0.view_Loss(com,vs[v], label, y_v)
            com_loss = loss0.criterion(y_p, label)
            
            loss = com_loss + 0.5*v_loss
            tot_loss.append(loss.item())
            y_pred = torch.argmax(y_p, dim=1)
            
            y_pred_vector.extend(y_pred)  
            labels_vector.extend(label) 
    acc = evaluate(labels_vector, y_pred_vector)
    print(f'Validation Epoch: {epoch+1:d}, Action Accuracy: {acc*100:.4f}% ,Test Loss: {np.mean(tot_loss)}'
          ,flush=True)  
    return acc, np.mean(tot_loss)
