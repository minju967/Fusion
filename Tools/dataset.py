from PIL import Image
from torchvision import transforms

import numpy as np
import torch.utils.data
import os
import torch
import pandas as pd
import random
import glob
import pyvista as pv
from sklearn.preprocessing import MinMaxScaler

class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, obj_path, nviews, shuffle=True):
        self.img_path = img_path
        self.obj_path = obj_path
        self.nview = nviews
        self.classes = os.listdir(self.img_path)
        self.target = {0:['A', 'C', 'E'], 1:['B', 'D']}  
        self.value = self.target.values()        
        self.mean = [0.46147138, 0.45751584, 0.44702336]
        self.std = [0.20240466, 0.19746633, 0.18430763]
        
        self.img_files, self.obj_files = [], []
        self.img_files.extend(path for path in sorted(glob.glob(self.img_path+'/*/*.png')))
        self.obj_files.extend(path for path in sorted(glob.glob(self.obj_path+'/*/*.obj')))

        # print(len(self.img_files)//self.nview)
        # print(len(self.obj_files))
        
        if shuffle == True:
            rand_idx = np.random.permutation(int(len(self.img_files)/self.nview))
            new_img_files, new_obj_files = [], []

            for i in range(len(rand_idx)):
                new_img_files.extend(self.img_files[rand_idx[i]*self.nview:(rand_idx[i]+1)*self.nview])
                obj_name = self.img_files[rand_idx[i]*self.nview].replace('_001.png','').split('/')[-1]+'.obj'
                obj_path = [p for p in self.obj_files if p.split('/')[-1] == obj_name]
                new_obj_files.extend(obj_path)

                if i == 0:
                    print(self.img_files[rand_idx[i]*self.nview])
                    print(obj_path[0])

            self.img_files = new_img_files
            self.obj_files = new_obj_files
             
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        self.df_objects = pd.DataFrame(columns=["object_name", "x_size", "y_size", "z_size", "n_cell", "n_point", "volume", "cutted_volume", \
                                   "x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "class"])
        
        for i, obj_p in enumerate(self.obj_files):
            object_name = obj_p.split('/')[-1]
            class_name = obj_p.split('/')[-2]
            if class_name in list(self.target.values())[0]:
                class_name = 'positive'
            else:       
                class_name = 'negative'
            mesh = pv.read(obj_p)
            cutted_size = pv.Box(mesh.bounds).volume - mesh.volume
        
            x_size = mesh.bounds[1] - mesh.bounds[0]
            y_size = mesh.bounds[3] - mesh.bounds[2]
            z_size = mesh.bounds[5] - mesh.bounds[4]
        
            self.df_objects.loc[i] = [object_name, x_size, y_size, z_size, mesh.n_cells, mesh.n_points, mesh.volume, cutted_size, \
                                    mesh.bounds[0], mesh.bounds[1], mesh.bounds[2], mesh.bounds[3], mesh.bounds[4], mesh.bounds[5], class_name]
        
        names = self.df_objects[['object_name']]
        classes = self.df_objects[['class']]

        self.new_df_objects = self.df_objects[["x_size", "y_size", "z_size", "n_cell", "n_point", "volume", "cutted_volume", \
                                   "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]]
        scalar = MinMaxScaler()
        scalar.fit(self.new_df_objects)
        scaled_objects = scalar.transform(self.new_df_objects)

        self.scaled_df_objects = pd.DataFrame(data=scaled_objects, columns=["x_size", "y_size", "z_size", "n_cell", "n_point", "volume", "cutted_volume", \
                                "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"])
                                
        self.scaled_df_objects['object_name'] = names
        self.scaled_df_objects['class'] = classes

    def __len__(self):
        return int(len(self.obj_files))

    def __getitem__(self, index):
        obj_path = self.obj_files[index]
        obj_name = obj_path.split('/')[-1]
        cls = obj_path.split('/')[-2]
        for i, val in enumerate(list(self.target.values())):
            if cls in val:
                class_id = list(self.target.keys())[i]

        # obj 13 elements data load
        target = self.scaled_df_objects.iloc[index]['class']
        if target == 'positive':
            target = 1
        else:
            target = 0 
        data = list(self.scaled_df_objects.iloc[index])[:-3]
        data = np.array(data)

        # obj Multi-view images data load
        imgs = []
        tf = transforms.ToTensor()
        for i in range(self.nview):
            im = Image.open(self.img_files[index*self.nview+i]).convert('RGB')
            if self.transform:
                im = tf(im)
                im = self.transform(im)
            imgs.append(im)

        return (obj_name, target, torch.stack(imgs), torch.tensor(data).float())