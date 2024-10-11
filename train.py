import torch
import yaml
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from data.dataset import generate_dataset
from data.data_utils import *
from model.models import WIN

def train():
    pl.seed_everything(1234)

    '''Training models.
    - train on carla: train_carla.yaml
    - train on kitti: train_kitti.yaml
    '''

    # choose the config for different datasets
    with open('./configs/train_carla.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device('cuda', config['cuda_index'][0])

    # generate model
    model = WIN(config).to(device)
    print(model.device)

    # generate dataset
    dataset = generate_dataset(config['dataset'], config)

    # split data in train and val
    len_data = dataset.__len__()
    len_train = int(len_data * config['train_val_split'][0])
    dataset_train, dataset_val = random_split(dataset, [len_train, len_data-len_train])

    dataloader_train = DataLoader(dataset_train, 
                                  batch_size=config['batch_size'], 
                                  num_workers=config['num_workers'],
                                  collate_fn=collate_fn_range_image,
                                  shuffle=True,
                                  pin_memory=True)
    
    dataloader_val = DataLoader(dataset_val, 
                                batch_size=config['batch_size'], 
                                num_workers=config['num_workers'],
                                collate_fn=collate_fn_range_image,
                                shuffle=False,
                                pin_memory=True)

    

    # training and validation
    checkpointcallback = ModelCheckpoint(monitor=config['monitor'],
                                         mode='min')
    
    # lightning trainer
    trainer = pl.Trainer(accelerator='gpu', 
                         devices=config['cuda_index'], 
                         max_epochs=config['max_epochs'], 
                         logger=TensorBoardLogger('train_model_logs'),
                         enable_checkpointing=True,
                         callbacks=[checkpointcallback],
                         fast_dev_run=False,
                         profiler='simple')

    if 'ckpt_path' in config:
        trainer.fit(model=model, 
                    train_dataloaders=dataloader_train, 
                    val_dataloaders=dataloader_val, 
                    ckpt_path=config['ckpt_path'])
    else :
        trainer.fit(model=model, 
                     train_dataloaders=dataloader_train, 
                     val_dataloaders=dataloader_val)

if __name__ == '__main__':
    train()
