import os
import argparse
import torch
from torch.nn.init import kaiming_uniform_
from datetime import datetime

from model.train import train_model
from model.dataloader import build_config

def main(args):
    run_id = args.dataset + '_' + args.traintype + '_' + datetime.today().strftime('%m-%d_%H%M')
    cfg = build_config(args.dataset,args.traintype,args.Dali)
            
    save_dir = os.path.join(cfg.saved_models_dir, run_id)  # type: ignore #
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)    
    train_model(cfg,save_dir, args, 0)
        
if __name__ == '__main__':

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    parser = argparse.ArgumentParser(description='Script to train Multi-label Classification model')
    parser.add_argument('--checkpoint', type=str, required=False, help='Path to the pre-trained model.')
    
    parser.add_argument("--traintype", dest='traintype', default='CS', choices=["CS", 'CV','CSet']
                        , required=False, help='Set train type from CS or CV or CSet')
    parser.add_argument("--loss_name", type=str, dest='loss', default='c', choices=['c','cv','cvl']
                            , required=False, help='Set loss from c or cv or cvl')
    parser.add_argument('--dataset', type=str, default='ntu-60', required=False, help='Dataset to use.', 
                        choices=["ntu-120", 'ntu-60',"pkummd", "n-ucla"])

    parser.add_argument("--device", dest='device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        required=False, help='Set CUDA_VISIBLE_DEVICES environment variable, optional')
    
    parser.add_argument('--batch_size', type=int, default=16, required=False, help='Batch size.')

    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs.')

    parser.add_argument('--num_workers', type=int, default=12, help='Number of workers in the dataloader.')

    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for the FC layers.')

    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay.')
    
    parser.add_argument("--temperature_f", type=float, default=0.7)
    
    parser.add_argument("--cache", type=str, default='true')
    
    parser.add_argument('--optimizer', type=str, default='ADAMW', choices=['ADAM', 'ADAMW', 'SGD'], help='provide optimizer preference')
    
    parser.add_argument('--validation_interval', type=int, default=10, help='Number of epochs between validation step.')

    parser.add_argument('--seed', type=int, default=7, help='Random seed.')
    
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    
    args = parser.parse_args()
    main(args)