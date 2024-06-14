#from functions import extract_LAI_from_RAS_file, explore_image, extract_all_LAI_from_RAS_file
import matplotlib.pyplot as plt
#import torch
import numpy as np
#datapath = './dataset/france/lai_ras/'
import random
import glob
import tifffile
filepaths = glob.glob('/home/luser/UniBw-STELAR/dataset/france2/processed_lai_npy/*.npy')
filepaths.sort()


import torch
device = torch.device("cuda")# if torch.cuda.is_available() else "cpu")

#labels_p = np.load('/home/luser/STELAR_Workbenches/saved_labels/lai_specific_france_mask_10_top.npy')
labels = np.load('/home/luser/UniBw-STELAR/dataset/test_saves/vista_labes_image.npy').astype(np.uint8)

#draw_time_index = random.randint(0, 245)
#selcted_time_files = filepaths[draw_time_index:draw_time_index+20]
#big_stack = torch.tensor([]).to(device)

for k in range(100):

    x_ind = random.choices(range(14), k=64)
    y_ind = random.choices(range(14), k=64)

    temporal_batches = torch.tensor([]).to(device)
    spatial_label_batches = torch.tensor([]).to(device)
    for i in range(len(x_ind)):
        time_strip_ind =  np.random.randint(0, len(filepaths)-65)
        considered_filepaths = filepaths[time_strip_ind:time_strip_ind+64]   
        temporal_strip = torch.tensor([]).to(device)

        x_coord = x_ind[i] * 64
        y_coord = y_ind[i] * 64
        
        space_label = labels[x_coord:x_coord+64, y_coord:y_coord+64]
        space_label = torch.from_numpy(space_label).to(device)

        spatial_label_batches = torch.cat((spatial_label_batches, space_label.unsqueeze(0)), axis=0)
        for filepath in considered_filepaths:
            numpy_array = np.load(filepath)
            torch_tensor = torch.from_numpy(numpy_array).to(device)
            numpy_array = 0.0

            #print(torch_tensor[x_coord:x_coord+64, y_coord:y_coord+64].shape)
            #temporal_strip.append(torch_tensor[x_coord:x_coord+64, y_coord:y_coord+64])
            temporal_strip = torch.cat((temporal_strip, torch_tensor[x_coord:x_coord+64, y_coord:y_coord+64].unsqueeze(0)), axis=0)
            #print("temporal_strip.shape", temporal_strip.shape)
        temporal_batches = torch.cat((temporal_batches, temporal_strip.unsqueeze(0)) , axis=0)

    print("temporal_batches.shape", temporal_batches.shape)
    print("spatial_label_batches.shape", spatial_label_batches.shape)

    array_lai = temporal_batches.cpu().numpy()
    tifffile.imsave('/home/luser/stelar_3d/storage/data_labels/train'+str(k)+'.tif', array_lai)


    array_lab = spatial_label_batches.cpu().numpy()
    tifffile.imsave('/home/luser/stelar_3d/storage/data_labels/lab'+str(k)+'.tif', array_lab)
