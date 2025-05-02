import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize      

class CLoss(nn.Module):
    def __init__(self, batch_size, device,num_actions,temperature=0.7, base_temperature=0.7):
        super(CLoss, self).__init__()
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.device=device
        self.num_actions=num_actions
        self.base_temperature = base_temperature
    def TBC_Loss(self, f_com, f_view, labels, p_labels, w0=None):
        batch_size = f_com.shape[0]
        
        labels = labels.contiguous().view(-1, 1)
        p_labels=p_labels.contiguous().view(-1, 1)
        f_com = f_com / torch.norm(f_com, dim=1, keepdim=True)
        f_view = f_view / torch.norm(f_view, dim=1, keepdim=True)
        anchor_dot_contrast = torch.matmul(f_com, f_view.T)/self.temperature

        mask_l = torch.eq(labels, labels.T)
        mask_l = mask_l.float().to(self.device)
        
        if w0 is None:
            w = torch.ones_like(mask_l).to(self.device) 
        else:
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=self.num_actions).to(self.device)
            one_hot_labels = one_hot_labels.squeeze(1).to(torch.float32)
        
            w = torch.matmul(one_hot_labels, w0.T)
        mask_p = mask_l * w
        
        all_one = torch.ones_like(mask_l).to(self.device)
        mask_n=(all_one-mask_l) * w
        
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        exp_logits = torch.exp(logits* torch.exp(mask_n))
        logits_exp_sum = torch.log(exp_logits.sum(1, keepdim=True) + logits)
        log_prob = (logits - logits_exp_sum)*mask_p

        if mask_p.sum() == 0:
            return 0
        mean_log_prob_pos = (log_prob).sum() / mask_p.sum()

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss /= batch_size**2
        if torch.isnan(loss):
            print(mask_n)
            a=exp_logits.sum(1, keepdim=True) + logits
            print('nan')
        return loss


