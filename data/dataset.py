import os
import torch
from torch.utils.data import Dataset
from data.data_utils import *
#from data.dataset_generator import register_dataset

"""
pytorch dataset for Carla, load low-resolution rangeimage and high-resolusion rangeimage(*.rimg)
the dataset can be downloaded from https://sgvr.kaist.ac.kr/~yskwon/papers/icra22-iln/carla.zip
"""

dataset_list = {}

def register_dataset(name):
    def decorator(cls):
        dataset_list[name] = cls
        return cls
    return decorator


def generate_dataset(dataset_name, dataset_args=None):
    """Generate a dataset depending on the dataset specifications."""
    dataset = dataset_list[dataset_name](dataset_args)
    return dataset


@register_dataset('Carla')
class CarlaDataset(Dataset):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # read the lidar specification
        lidar_path = os.path.join(config['data_dir'], 'lidar_specification.yaml')
        self.lidar_out = initialize_lidar(lidar_path)
        self.lidar_out['channels'] = int(config['res_out'].split('_')[0])
        self.lidar_out['points_per_ring'] = int(config['res_out'].split('_')[1])

        self.lidar_in = copy.deepcopy(self.lidar_out)
        self.lidar_in['channels'] = int(config['res_in'].split('_')[0])
        self.lidar_in['points_per_ring'] = int(config['res_in'].split('_')[1])

        # get all the file by their absolute path
        gt_file_path_list = []
        input_file_path_list = []

        if config['status'] == 'train':
            for scene in config['train_sceneIDs']:
                gt_file_dir = os.path.join(config['data_dir'], scene, config['res_out']) 
                for _, _, filenames in sorted(os.walk(gt_file_dir)):
                    for filename in sorted(filenames):
                        file_path = os.path.join(gt_file_dir, filename)
                        gt_file_path_list.append(file_path)
            for scene in config['train_sceneIDs']:
                input_file_dir = os.path.join(config['data_dir'], scene, config['res_in'])
                for _, _, filenames in sorted(os.walk(input_file_dir)):
                    for filename in sorted(filenames):
                        file_path = os.path.join(input_file_dir, filename)
                        input_file_path_list.append(file_path)
        else:
            for scene in config['test_sceneIDs']:
                gt_file_dir = os.path.join(config['data_dir'], scene, config['res_out']) 
                for _, _, filenames in sorted(os.walk(gt_file_dir)):
                    for filename in sorted(filenames):
                        file_path = os.path.join(gt_file_dir, filename)
                        gt_file_path_list.append(file_path)

            for scene in config['test_sceneIDs']:
                input_file_dir = os.path.join(config['data_dir'], scene, config['res_in']) 
                for _, _, filenames in sorted(os.walk(input_file_dir)):
                    for filename in sorted(filenames):
                        file_path = os.path.join(input_file_dir, filename)
                        input_file_path_list.append(file_path)

        self.gt_file_path_list = gt_file_path_list
        self.input_file_path_list = input_file_path_list

    def __getitem__(self, index):
        gt_file_path = self.gt_file_path_list[index]
        

        # high resolution 
        range_image_hr = read_range_image_binary(file_path=gt_file_path, lidar=self.lidar_out) / float(self.lidar_in['norm_r'])

        # low resolution
        if self.config['down_sample']:
            range_image_lr = downsample_range_image(range_image_hr, downsample_rate=self.config['up_factor'])
            # tansfer to (d,x,y,z)
            self.lidar_in = downsample_lidar(self.lidar_out, self.config['up_factor'])
            points_lr = range_image_to_points(range_image_lr, lidar=self.lidar_in)
            image_lr = points_to_image(points=points_lr, lidar=self.lidar_in)
        else:
            input_file_path = self.input_file_path_list[index]
            range_image_lr = read_range_image_binary(file_path=input_file_path, lidar=self.lidar_out) / float(self.lidar_in['norm_r'])
            # tansfer to (d,x,y,z)
            points_lr = range_image_to_points(range_image_lr, lidar=self.lidar_in)
            image_lr = points_to_image(points=points_lr, lidar=self.lidar_in)
            
        return image_lr, torch.unsqueeze(range_image_hr, dim=0)
    
    def __len__(self):
        return len(self.gt_file_path_list)

"""
pytorch dataset for Kitti, load low-resolution rangeimage and high-resolusion rangeimage(*.rimg)
the dataset can be downloaded from 
"""

@register_dataset('Kitti')
class KittiDataset(Dataset):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config
        
        # read the lidar specification
        lidar_path = os.path.join(config['data_dir'], 'lidar_specification.yaml')
        self.lidar_out = initialize_lidar(lidar_path)
        self.lidar_out['channels'] = int(config['res_out'].split('_')[0])
        self.lidar_out['points_per_ring'] = int(config['res_out'].split('_')[1])
        self.lidar_in = copy.deepcopy(self.lidar_out)
        self.lidar_in['channels'] = int(config['res_in'].split('_')[0])
        self.lidar_in['points_per_ring'] = int(config['res_in'].split('_')[1])

        # get all the file by their absolute path

        gt_file_path_list = []
        if config['status'] == 'train':
            gt_file_dir = os.path.join(config['data_dir'], 'train_')
            for _, _, filenames in os.walk(gt_file_dir):
                for filename in sorted(filenames):
                    file_path = os.path.join(gt_file_dir, filename)
                    gt_file_path_list.append(file_path)
        else:
            gt_file_dir = os.path.join(config['data_dir'], 'val_')
            for root, _, filenames in os.walk(gt_file_dir):
                for filename in sorted(filenames):
                    file_path = os.path.join(gt_file_dir, filename)
                    gt_file_path_list.append(file_path)

        self.gt_file_path_list = gt_file_path_list


    
    
    def __getitem__(self, index):
        gt_file_path = self.gt_file_path_list[index]

        # high resolution 
        range_xyz_image_hr = (read_range_xyz_image_npy(file_path=gt_file_path) / float(self.lidar_in['norm_r']))
        range_image_hr = range_xyz_image_hr[..., 0] 

        # low resolution
        range_image_lr = downsample_range_image(range_image_hr, downsample_rate=self.config['up_factor'])
        
        self.lidar_in = downsample_lidar(self.lidar_out, self.config['up_factor'])
        points_lr = range_image_to_points(range_image_lr, lidar=self.lidar_in)
        image_lr = points_to_image(points=points_lr, lidar=self.lidar_in)
        return image_lr, torch.unsqueeze(range_image_hr, dim=0)

    def __len__(self):
        return len(self.gt_file_path_list)
