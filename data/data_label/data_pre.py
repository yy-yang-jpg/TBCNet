import pandas as pd
import os
import torch
import h5py
import cv2
import threading
from torchvision.transforms import (
    Resize,   
)
def make_index(annotations,dataset):
    
    data = []
    data_dfview = {}
    data_saview={}            
    with open(annotations, 'r') as annotations_file:
        next(annotations_file)  # Skip the first line
        for count, row in enumerate(annotations_file):
            if dataset == 'ntu-120' or dataset == 'ntu-60':
                video_avi, subject, action, viewpoint,r,setting = row.split(',') 
                setting=setting.strip('\n')      
            elif dataset == 'pkummd':
                video_avi, subject, action, viewpoint,_,setting = row.split(',') 
                r=video_avi[0]
                setting=setting.strip('\n')      
            video_id=video_avi.split('.')[0]
            data.append([video_id, action, viewpoint,subject,r,setting])
            if f"{action}_{subject}_{r}_{setting}" not in data_dfview.keys():
                data_dfview[f"{action}_{subject}_{r}_{setting}"] = []
            data_dfview[f"{action}_{subject}_{r}_{setting}"].append(video_id)
    L=[]        
    for viewpoint in data_dfview.keys():
        if len(data_dfview[viewpoint])!=3:
            L.append(viewpoint)
            print(data_dfview[viewpoint])
    out=[]
    for i in L:
        out.extend(data_dfview[i])
    print(out)
    return data,data_dfview,data_saview
    

def indexNTU60_cs(path,save_path):
    
    train_df_rows = []
    test_df_rows = []
    train = ['P001', 'P002', 'P004', 'P005', 'P008', 'P009', 'P013', 'P014', 'P015', 'P016', 'P017', 'P018', 'P019',
         'P025', 'P027', 'P028', 'P031', 'P034', 'P035', 'P038']
    rows=os.listdir(path)
    for row in rows:
        s_num, cam_id, sub_id, rep_num, act_id = row[0:4], row[4:8], row[8:12], row[12:16], row[16:20]
        if sub_id in train:
            train_df_rows.append([row, int(sub_id[1:]),int( act_id[1:]),int( cam_id[1:]),int( rep_num[1:]),int( s_num[1:])])
        else:
            test_df_rows.append([row, int(sub_id[1:]),int( act_id[1:]),int( cam_id[1:]),int( rep_num[1:]),int( s_num[1:])])
            
    df = pd.DataFrame(train_df_rows, columns=['video_id', 'subject', 'action', 'camera', 'repetition', 'setup'])
    df.to_csv(save_path+"NTU60Train_CS.csv", index=False)
    
    df = pd.DataFrame(test_df_rows, columns=['video_id', 'subject','action', 'camera',  'repetition', 'setup'])
    df.to_csv(save_path+"NTU60Test_CS.csv", index=False)
    
def indexNTU60_cv(path,save_path):
    train_df_rows = []
    test_df_rows = []
    train = ['C002', 'C003']
    for row in os.listdir(path):
        s_num, cam_id, sub_id, rep_num, act_id = row[0:4], row[4:8], row[8:12], row[12:16], row[16:20]
        if cam_id in train:
            train_df_rows.append([row,int( sub_id[1:]),int( act_id[1:]),int( cam_id[1:]),int( rep_num[1:]),int( s_num[1:])])
        else:
            test_df_rows.append([row, int(sub_id[1:]),int( act_id[1:]),int( cam_id[1:]),int( rep_num[1:]),int( s_num[1:])])
            
    df = pd.DataFrame(train_df_rows, columns=['video_id', 'subject', 'action', 'camera', 'repetition', 'setup'])
    df.to_csv(save_path+"NTU60Train_CV.csv", index=False)
    
    df = pd.DataFrame(test_df_rows, columns=['video_id', 'subject', 'action', 'camera', 'repetition', 'setup'])
    df.to_csv(save_path+"NTU60Test_CV.csv", index=False)   
    
def indexNTU120_cs(path,save_path):
    train_df_rows = []
    test_df_rows = []
    train1 = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59]
    train2 = [70, 74, 78,80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103]
    for row in os.listdir(path+'ntu-60/'):
        s_num, cam_id, sub_id, rep_num, act_id = row[0:4], row[4:8], row[8:12], row[12:16], row[16:20]
        if int(sub_id[1:]) in train1 or int(sub_id[1:]) in train2:
            train_df_rows.append(['ntu-60/'+row, int(sub_id[1:]),int(act_id[1:]),int(cam_id[1:]),int(rep_num[1:]),int(s_num[1:])])
        else:
            test_df_rows.append(['ntu-60/'+row, int(sub_id[1:]),int(act_id[1:]),int(cam_id[1:]),int(rep_num[1:]),int(s_num[1:])])
            
    for row in os.listdir(path+'ntu-120/'):
        s_num, cam_id, sub_id, rep_num, act_id = row[0:4], row[4:8], row[8:12], row[12:16], row[16:20]
        if int(sub_id[1:]) in train1 or int(sub_id[1:]) in train2:
            train_df_rows.append(['ntu-120/'+row, int(sub_id[1:]),int(act_id[1:]),int(cam_id[1:]),int( rep_num[1:]),int( s_num[1:])])
        else:
            test_df_rows.append(['ntu-120/'+row, int(sub_id[1:]),int(act_id[1:]),int( cam_id[1:]),int( rep_num[1:]),int( s_num[1:])])
                

    df = pd.DataFrame(train_df_rows, columns=['video_id', 'subject', 'action', 'camera', 'repetition', 'setup'])
    df.to_csv(save_path+"NTUTrain_CS.csv", index=False)
    
    df = pd.DataFrame(test_df_rows, columns=['video_id', 'subject', 'action', 'camera', 'repetition', 'setup'])
    df.to_csv(save_path+"NTUTest_CS.csv", index=False)
    
def indexNTU120_cv(path,save_path):
    train_df_rows = []
    test_df_rows = []
    
    for video in os.listdir(path+'ntu-60/'):
        if int(video[1:4]) % 2 == 0:
            train_df_rows.append([video, int(video[9:12]), int(video[17:20]), int(video[5:8]), int(video[13:16]), int(video[1:4])])
        else:
            test_df_rows.append([video, int(video[9:12]), int(video[17:20]), int(video[5:8]), int(video[13:16]), int(video[1:4])])
    for video in os.listdir(path+'ntu-120/'):
        if int(video[1:4]) % 2 == 0:
            train_df_rows.append([video, int(video[9:12]), int(video[17:20]), int(video[5:8]), int(video[13:16]), int(video[1:4])])
        else:
            test_df_rows.append([video, int(video[9:12]), int(video[17:20]), int(video[5:8]), int(video[13:16]), int(video[1:4])])

    df = pd.DataFrame(train_df_rows, columns=['video_id', 'subject', 'action', 'camera', 'repetition', 'setup'])
    
    df = pd.DataFrame(test_df_rows, columns=['video_id', 'subject', 'action', 'camera', 'repetition', 'setup'])
    df.to_csv(save_path+"NTUTest_CV.csv", index=False)    
    
       

def video_to_hdf5(video_path,save_path, video_names):
    resize = Resize([480, 270])
    for video_name in video_names:
        name=video_name.split('.')[0]
        if os.path.exists(f'{save_path}{name}.hdf5'):
            pass
        else:
            cap = cv2.VideoCapture(os.path.join(video_path, video_name))
            
            frames = []
            ret, frame = cap.read()
            while ret:
                frame = torch.as_tensor(frame)
                frame = frame.permute(2, 0, 1)
                frame = resize(frame)
                frames.append(frame)
                
                ret, frame = cap.read()
            tframes = torch.stack([frame for frame in frames])
            frames.clear()
            
            with h5py.File(f'{save_path}{name}.hdf5', 'w') as f:
                    dset = f.create_dataset('default', data=tframes)
            del tframes
def process_videos_in_parallel(video_path, save_path, num_threads=4):
    video_names = os.listdir(video_path)
    for i in video_names:
        if i.endswith('.txt') or i.endswith('.txtnew'):
            video_names.remove(i)
    video_sublists = [video_names[i:i + len(video_names) // num_threads] for i in range(0, len(video_names), len(video_names) // num_threads)]

    threads = []
    for sublist in video_sublists:
        thread = threading.Thread(target=video_to_hdf5, args=(video_path, save_path, sublist))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()    
    
    
if __name__ == '__main__':
    data_path = ".\data\dataset"
    data_names =['ntu-rgb/ntu-120/', 'ntu-rgb/ntu-60/'] 
    num_threads = 10 
    for data_name in data_names:
        path=data_path+data_name
        save_path = ".\data\dataset_hdf5\\"+data_name   
        os.makedirs(save_path,exist_ok=True)
        process_videos_in_parallel(path, save_path, num_threads) 