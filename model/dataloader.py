import os
import math
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (
    Normalize
)
import h5py
import warnings
warnings.filterwarnings("ignore")

def build_config(dataset,clabel):
    cfg = type('', (), {})()
    if dataset == 'ntu-120':
        cfg.videos_folder =  f'./data/dataset_hdf5/ntu-rgb/' # type: ignore
        cfg.train_annotations = f'./data/NTUTrain_{clabel}.csv' # type: ignore
        cfg.test_annotations = f'./data/NTUTest_{clabel}.csv' # type: ignore
        cfg.num_actions = 120 # type: ignore
    elif dataset == 'ntu-60':
        cfg.videos_folder =  f'./data/dataset_hdf5/ntu-rgb/' # type: ignore
        cfg.train_annotations = f'./data/NTU60Train_{clabel}.csv' # type: ignore
        cfg.test_annotations = f'./data/NTU60Test_{clabel}.csv' # type: ignore
        cfg.num_actions = 60 # type: ignore
    elif dataset == 'n-ucla':
        cfg.videos_folder =  f'C:/Users/Yyy/Desktop/data/hdf5/n-ucla/' # type: ignore
        cfg.train_annotations = f"./data/NUCLATrain_{clabel}.csv" # type: ignore
        cfg.test_annotations = f"./data/NUCLATest_{clabel}.csv" # type: ignore
        cfg.num_actions = 10 # type: ignore
    elif dataset == "pkummd":
        cfg.videos_folder =  f'/home/yyy/data{folder_type}/pku-mmd/' # type: ignore
        cfg.train_annotations = f'./data/PKUTrain_{clabel}0.csv' # type: ignore
        cfg.test_annotations = f'./data/PKUTest_{clabel}00.csv' # type: ignore
        cfg.num_actions = 51 # type: ignore
    else:
        raise NotImplementedError
    
    cfg.clabel =clabel  # type: ignore
    cfg.dataset = dataset # type: ignore
    cfg.saved_models_dir  = './results/saved_models' # type: ignore
    cfg.outputs_folder = './results/outputs' # type: ignore
    cfg.tf_logs_dir = './results/logs' # type: ignore
    return cfg

def frame_creation(video_id, videos_folder, num_frames, transform,video_type='.hdf5'):


    video_path = os.path.join(videos_folder, video_id) + video_type
    with h5py.File(video_path, 'r') as frames0:
        #frames_out =  torch.from_numpy(frames0['default'][:]/255).float() # type: ignore 
        frames_out_np = frames0['default'][:]/255
    frames_out = torch.tensor(frames_out_np, dtype=torch.float32)#, device='cuda:0')
    # tchw
    if transform:
        frames_out0 = frames_out.transpose(0, 1) #CTHW
        frames_out1 = transform(frames_out0)
        frames_out = frames_out1.transpose(0, 1) #tchw
    if frames_out.shape[0] < 16:
        padding = frames_out[-1].unsqueeze(0).repeat(16 - frames_out.shape[0], 1, 1, 1)
        frames_out = torch.cat((frames_out, padding), dim=0)
    return frames_out
    
class MVA_Dataset(Dataset):

    def __init__(self, cfg, data_split, transform=None,view_num=3, height=270, width=480):
        super(MVA_Dataset,self).__init__()
        self.view_num = view_num
        self.dataset = cfg.dataset
        self.data_split = data_split 
        self.videos_folder = cfg.videos_folder
        self.height = height
        self.width = width
        self.transform = transform
        self.num_frames = 16

        if data_split == "train":
           self.annotations = cfg.train_annotations
           if cfg.clabel =='CV':
               self.view_num = 2
        else:
           self.annotations = cfg.test_annotations
           if cfg.clabel =='CV':
               self.view_num = 1
                
        self.data = []
        self.data_dfview = {}
          
        with open(self.annotations, 'r') as annotations_file:
            next(annotations_file)  # Skip the first line
            for count, row in enumerate(annotations_file):
                if self.dataset == 'n-ucla':
                    video_avi, subject, action, viewpoint = row.split(',')   
                    viewpoint=viewpoint.strip('\n')
                    setting=(video_avi.split('.')[0])[-2:]
                    r=0
                elif self.dataset == 'ntu-120' or self.dataset == 'ntu-60':
                    video_avi, subject, action, viewpoint,r,setting = row.split(',') 
                    setting=setting.strip('\n')      
                elif self.dataset == 'pkummd':
                    video_avi, subject, action, viewpoint,_,setting = row.split(',') 
                    r=video_avi[0]
                    setting=setting.strip('\n')      
                video_id=video_avi.split('.')[0]
                
                self.data.append([video_id, action, viewpoint,subject,r,setting])
                if f"{action}_{subject}_{r}_{setting}" not in self.data_dfview.keys():
                    self.data_dfview[f"{action}_{subject}_{r}_{setting}"] = []
                self.data_dfview[f"{action}_{subject}_{r}_{setting}"].append(video_id)
        self.data_saview={}      
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        input_l=[]
        anchor = self.data[index]
        #sa_view = random.choice(self.data_saview)
        video_id, action, subject,r,setting = anchor[0], anchor[1],anchor[3], anchor[4], anchor[5]
        data_dfview= self.data_dfview[f"{action}_{subject}_{r}_{setting}"]

        for video_id in data_dfview:
            video = frame_creation(video_id, self.videos_folder, self.num_frames, self.transform)
            if video.shape[0] != 16:
                padding_needed = max(0, 16 - video.shape[0])
                padding = torch.stack([video[-1,:,:,:]]*padding_needed,dim=0)

                video = torch.cat((video, padding), dim=0)
            input_l.append(video)
            
        if len(input_l)==1:
            input_l.append(input_l[-1])
        input=torch.stack(input_l,dim=0)
        
        return input, int(action)


class Normalize0(Normalize):
    """
    Normalize the (CTHW) video clip by mean subtraction and division by standard deviation

    Args:
        mean (3-tuple): pixel RGB mean
        std (3-tuple): pixel RGB standard deviation
        inplace (boolean): whether do in-place normalization
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        vid = x.permute(1, 0, 2, 3)  # C T H W to T C H W
        vid = super().forward(vid)
        vid = vid.permute(1, 0, 2, 3)  # T C H W to C T H W
        return vid


class RandomShortSideScale(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.short_side_scale``. The size
    parameter is chosen randomly in [min_size, max_size].
    """

    def __init__(
        self,
        min_size: int,
        max_size: int,
        interpolation: str = "bilinear",
        backend: str = "pytorch",
    ):
        super().__init__()
        self._min_size = min_size
        self._max_size = max_size
        self._interpolation = interpolation
        self._backend = backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        size = torch.randint(self._min_size, self._max_size + 1, (1,)).item()
        return short_side_scale(
            x, int(size), self._interpolation, self._backend 
        )

class ShortSideScale(torch.nn.Module):
    """
    ``nn.Module`` wrapper for ``pytorchvideo.transforms.functional.short_side_scale``.
    """

    def __init__(
        self, size: int, interpolation: str = "bilinear", backend: str = "pytorch"
    ):
        super().__init__()
        self._size = size
        self._interpolation = interpolation
        self._backend = backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor with shape (C, T, H, W).
        """
        return short_side_scale(
            x, self._size, self._interpolation, self._backend
        )

def short_side_scale(
    x: torch.Tensor,
    size: int,
    interpolation: str = "bilinear",
    backend: str = "pytorch",
) -> torch.Tensor:
    """
    """  # noqa
    assert len(x.shape) == 4
    assert x.dtype == torch.float32
    assert backend in ("pytorch", "opencv")
    c, t, h, w = x.shape
    if w < h:
        new_h = int(math.floor((float(h) / w) * size))
        new_w = size
    else:
        new_h = size
        new_w = int(math.floor((float(w) / h) * size))
    if backend == "pytorch":
        return torch.nn.functional.interpolate(
            x, size=(new_h, new_w), mode=interpolation, align_corners=False
        )
    else:
        raise NotImplementedError(f"{backend} backend not supported.")