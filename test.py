import torch
import yaml
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from data.dataset import generate_dataset
from data.data_utils import *
from model.models import WIN
import argparse


def test(args):
    pl.seed_everything(1234)
    
    # choose the config for different datasets

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device('cuda', config['cuda_index'][0])
    
    # generate model
    model = WIN(config).to(device)
    print(model.device)

    # generate dataset
    dataset = generate_dataset(config['dataset'], config)

    dataloader_test = DataLoader(dataset, 
                                  batch_size=config['batch_size'], 
                                  num_workers=config['num_workers'],
                                  collate_fn=collate_fn_range_image,
                                  shuffle=False,
                                  pin_memory=True)

    # testing
    checkpointcallback = ModelCheckpoint(monitor=config['monitor'],
                                         mode='min')
    
    trainer = pl.Trainer(accelerator='gpu', 
                         devices=config['cuda_index'], 
                         max_epochs=config['max_epochs'], 
                         logger=TensorBoardLogger('test_model_logs'),
                         enable_checkpointing=True,
                         callbacks=[checkpointcallback],
                         fast_dev_run=False)

    trainer.test(model=model, dataloaders=dataloader_test, ckpt_path=config['ckpt_path'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration')
    parser.add_argument('--config', default='./configs/test_carla.yaml', help='config for testing')
    args = parser.parse_args()
    test(args)
