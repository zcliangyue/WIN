import os
import torch
import torch.nn as nn
import torch.cuda as cu
import pytorch_lightning as pl
from torchmetrics import MeanMetric, MinMetric
from functools import partial
from typing import Any, Tuple

from model.components.iln import *
from metrics.metrics import *
from data.data_utils import *
from model.components.lsr import lsr
from model.components.tulip import TULIP
from model.components.liif import LIIFLiDAR
from torch.optim.lr_scheduler import MultiStepLR
from model.components.edsr_lidar import EDSRLiDAR, ConvLiDAR
from model.model_generator import register_model, generate_model

'''The implement of all the related models. In order to integrate all models into the same training framework, we repackaged all models but kept the hyperparameters and model settings. 
In addition, the inputs to all model classes are configuration information, which allows us to simply generate the corresponding model based on the configuration information.
'''


@register_model('lsr')
class lsr_model(nn.Module):
    '''The model provided by TULIP.
    We encapsulate it to articulate into our code. Our output is 'd, d.detach(), torch.zeros_like(d)', which allows us to train a single-branch network using the WIN class.
    '''
    def __init__(self,
                 config
                 ):
        super().__init__()

        # initial the params of backbone
        lidar_path = os.path.join(config['data_dir'], 'lidar_specification.yaml')
        self.lidar_out = initialize_lidar(lidar_path)
        channels_out = int(config['res_out'].split('_')[0])
        points_per_ring_out = int(config['res_out'].split('_')[1])
        self.lidar_out['channels'] = channels_out
        self.lidar_out['points_per_ring'] = points_per_ring_out

        self.lidar_in = copy.deepcopy(self.lidar_out)
        self.lidar_in['channels'] = int(config['res_in'].split('_')[0])
        self.lidar_in['points_per_ring'] = int(config['res_in'].split('_')[1])
        
    
        # backbone
        self.net = lsr(scale_h=4, scale_w=1)


    def forward(self, batch_dict):

        input = batch_dict['image_lr_batch'][:, [0], :, :]
        d = self.net(input)
            
        return d, d, torch.zeros_like(d)
    

@register_model('tulip_base')
class TULIP_base(nn.Module):
    '''The base model provided by TULIP.
    We encapsulate it to articulate into our code. Our output is 'd, d.detach(), torch.zeros_like(d)', which allows us to train a single-branch network using the WIN class.
    For the hyperparameters of the network, we are consistent with the source code provided by TULIP.
    '''
    def __init__(self,
                 config
                 ):
        super().__init__()

        # initial the params of backbone
        lidar_path = os.path.join(config['data_dir'], 'lidar_specification.yaml')
        self.lidar_out = initialize_lidar(lidar_path)
        channels_out = int(config['res_out'].split('_')[0])
        points_per_ring_out = int(config['res_out'].split('_')[1])
        self.lidar_out['channels'] = channels_out
        self.lidar_out['points_per_ring'] = points_per_ring_out

        self.lidar_in = copy.deepcopy(self.lidar_out)
        self.lidar_in['channels'] = int(config['res_in'].split('_')[0])
        self.lidar_in['points_per_ring'] = int(config['res_in'].split('_')[1])
        
        # backbone
        self.net = TULIP(img_size=(self.lidar_in['channels'], self.lidar_in['points_per_ring']),
                         target_img_size=(self.lidar_out['channels'], self.lidar_out['points_per_ring']),
                         pixel_shuffle=True,
                         circular_padding=True,
                         log_transform=True,
                         patch_size=(1, 4),
                         window_size=(2, 8),depths=(2, 2, 2, 2),
                         num_heads=(3, 6, 12, 24),
                         norm_layer=partial(nn.LayerNorm, eps=1e-6))
        

    def forward(self, batch_dict):

        input = batch_dict['image_lr_batch'][:, [0], :, :]
        d = self.net(input)

        return d, d.detach(), torch.zeros_like(d)
    
    
@register_model('LIIF')
class LIIF(nn.Module):
    '''The model provided by LIIF.
    We encapsulate it to articulate into our code. Our output is 'd, d.detach(), torch.zeros_like(d)', which allows us to train a single-branch network using the WIN class.
    '''
    def __init__(self,
                 config
                 ):
        super().__init__()

        lidar_path = os.path.join(config['data_dir'], 'lidar_specification.yaml')
        self.lidar_out = initialize_lidar(lidar_path)
        channels_out = int(config['res_out'].split('_')[0])
        points_per_ring_out = int(config['res_out'].split('_')[1])
        self.lidar_out['channels'] = channels_out
        self.lidar_out['points_per_ring'] = points_per_ring_out

        if(config['down_sample']):  
            self.lidar_in = downsample_lidar(self.lidar_out, config['up_factor'])
        else:
            self.lidar_in = copy.deepcopy(self.lidar_out)
            self.lidar_in['channels'] = int(config['res_in'].split('_')[0])
            self.lidar_in['points_per_ring'] = int(config['res_in'].split('_')[1])
        
        self.coord = generate_laser_directions(self.lidar_out)
        self.coord = torch.unsqueeze(torch.tensor(normalization_queries(self.coord, self.lidar_in)), 0).cuda()
        
        self.net = LIIFLiDAR()
        

    def forward(self, batch_dict):

        batch_size = batch_dict['image_lr_batch'].shape[0]
        d = self.net(batch_dict['image_lr_batch'], self.coord.repeat(batch_size, 1, 1))
        d = d.reshape(batch_size, 1, self.lidar_out['channels'], self.lidar_out['points_per_ring']) # [B, 1, H, W]

        return d, d.detach(), torch.zeros_like(d)


@register_model('ILN')
class ILN_model(nn.Module):
    '''The model provided by ILN.
    We encapsulate it to articulate into our code. Our output is 'd, d.detach(), torch.zeros_like(d)', which allows us to train a single-branch network using the WIN class.
    '''
    def __init__(self,
                 config
                 ):
        super().__init__()

         # initial the params of backbone
        lidar_path = os.path.join(config['data_dir'], 'lidar_specification.yaml')
        self.lidar_out = initialize_lidar(lidar_path)
        channels_out = int(config['res_out'].split('_')[0])
        points_per_ring_out = int(config['res_out'].split('_')[1])
        self.lidar_out['channels'] = channels_out
        self.lidar_out['points_per_ring'] = points_per_ring_out

        if(config['down_sample']):  
            self.lidar_in = downsample_lidar(self.lidar_out, config['up_factor'])
        else:
            self.lidar_in = copy.deepcopy(self.lidar_out)
            self.lidar_in['channels'] = int(config['res_in'].split('_')[0])
            self.lidar_in['points_per_ring'] = int(config['res_in'].split('_')[1])
        
        self.coord = generate_laser_directions(self.lidar_out)
        self.coord = torch.unsqueeze(torch.tensor(normalization_queries(self.coord, self.lidar_in)), 0).cuda()
        

        # backbone
        self.rv_encoder = EDSRLiDAR(n_resblocks=16, n_feats=64, res_scale=1.0, use_xyz=config['use_xyz'])
        self.inter = ILN_Interpolate(dim=64)
        

    def forward(self, batch_dict):
        
        batch_size = batch_dict['image_lr_batch'].shape[0]

        feat = self.rv_encoder(batch_dict['image_lr_batch'])
        d = self.inter(batch_dict['image_lr_batch'], feat, self.coord.repeat(batch_size, 1, 1))

        d = d.reshape(batch_size, 1, self.lidar_out['channels'], self.lidar_out['points_per_ring']) # [B, 1, H, W]

        return d, d.detach(), torch.zeros_like(d)
    
@register_model('ILN_xy')
class ILN_xy(nn.Module):
    '''The model for ablation.
    It only performs interpolation on HRV.
    '''
    def __init__(self,
                 config
                 ):
        super().__init__()

         # initial the params of backbone
        lidar_path = os.path.join(config['data_dir'], 'lidar_specification.yaml')
        self.lidar_out = initialize_lidar(lidar_path)
        channels_out = int(config['res_out'].split('_')[0])
        points_per_ring_out = int(config['res_out'].split('_')[1])
        self.lidar_out['channels'] = channels_out
        self.lidar_out['points_per_ring'] = points_per_ring_out

        if(config['down_sample']):  
            self.lidar_in = downsample_lidar(self.lidar_out, config['up_factor'])
        else:
            self.lidar_in = copy.deepcopy(self.lidar_out)
            self.lidar_in['channels'] = int(config['res_in'].split('_')[0])
            self.lidar_in['points_per_ring'] = int(config['res_in'].split('_')[1])
        
        self.coord = generate_laser_directions(self.lidar_out)
        self.coord = torch.unsqueeze(torch.tensor(normalization_queries(self.coord, self.lidar_in)), 0)
        


        # backbone
        self.rv_encoder = EDSRLiDAR(n_resblocks=16, n_feats=64, res_scale=1.0, use_xyz=config['use_xyz'])
        self.inter = interpolate_xy()
        
    def forward(self, batch_dict):
        
        batch_size = batch_dict['image_lr_batch'].shape[0]

        feat = self.rv_encoder(batch_dict['image_lr_batch'])
        xy = self.inter(batch_dict['image_lr_batch'], feat, self.coord.repeat(batch_size, 1, 1))

        xy = xy.reshape(batch_size, 1, self.lidar_out['channels'], self.lidar_out['points_per_ring']) # [B, 1, H, W]

        # xy to depth
        # v_dir = torch.unsqueeze(torch.cos(torch.linspace(start=self.lidar_out['min_v'], end=self.lidar_out['max_v'], steps=self.lidar_out['channels'])), dim=1)
        v_dir = torch.unsqueeze(torch.cos(torch.tensor(zenith[::-1].copy(), dtype=torch.float32)), dim=1)
        v_dir = v_dir.repeat(1, self.lidar_out['points_per_ring'])
        v_dir = v_dir.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        d = xy / v_dir.cuda()

        return d, d.detach(), torch.zeros_like(d)

@register_model('ILN_z')
class ILN_z(nn.Module):
    '''The model for ablation.
    It only performs interpolation on VRV.
    '''
    def __init__(self,
                 config
                 ):
        super().__init__()

        # initial the params of backbone
        lidar_path = os.path.join(config['data_dir'], 'lidar_specification.yaml')
        self.lidar_out = initialize_lidar(lidar_path)
        channels_out = int(config['res_out'].split('_')[0])
        points_per_ring_out = int(config['res_out'].split('_')[1])
        self.lidar_out['channels'] = channels_out
        self.lidar_out['points_per_ring'] = points_per_ring_out

        if(config['down_sample']):  
            self.lidar_in = downsample_lidar(self.lidar_out, config['up_factor'])
        else:
            self.lidar_in = copy.deepcopy(self.lidar_out)
            self.lidar_in['channels'] = int(config['res_in'].split('_')[0])
            self.lidar_in['points_per_ring'] = int(config['res_in'].split('_')[1])
        
        self.coord = generate_laser_directions(self.lidar_out)
        self.coord = torch.unsqueeze(torch.tensor(normalization_queries(self.coord, self.lidar_in)), 0)
        


        # backbone
        self.rv_encoder = EDSRLiDAR(n_resblocks=16, n_feats=64, res_scale=1.0, use_xyz=config['use_xyz'])
        self.inter = interpolate_z(dim=64)

        

    def forward(self, batch_dict):
        
        batch_size = batch_dict['image_lr_batch'].shape[0]

        feat = self.rv_encoder(batch_dict['image_lr_batch'][:, [0], :, :])
        z = self.inter(batch_dict['image_lr_batch'], feat, self.coord.repeat(batch_size, 1, 1))

        z = z.reshape(batch_size, 1, self.lidar_out['channels'], self.lidar_out['points_per_ring'])

        # z to depth
        v_dir = torch.unsqueeze(torch.sin(torch.linspace(start=self.lidar_out['min_v'], end=self.lidar_out['max_v'], steps=self.lidar_out['channels'])), dim=1)
        v_dir = v_dir.repeat(1, self.lidar_out['points_per_ring'])
        v_dir = v_dir.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        d = z / v_dir.cuda()

        return d, d, torch.zeros_like(d)


@register_model('WIN')
class WIN_net(nn.Module):
    '''The model of our WIN.
    We output the interpolation results for both branches(d_xy, d_z), as well as the prediction results from the CSM module(w). 
    Further loss calculations will be integrated in the lightning module 'WIN'.
    '''
    def __init__(self,
                 config
                 ):
        super().__init__()

        # initial the params of backbone
        self.dataset = config['dataset']
        lidar_path = os.path.join(config['data_dir'], 'lidar_specification.yaml')
        self.lidar_out = initialize_lidar(lidar_path)
        channels_out = int(config['res_out'].split('_')[0])
        points_per_ring_out = int(config['res_out'].split('_')[1])
        self.lidar_out['channels'] = channels_out
        self.lidar_out['points_per_ring'] = points_per_ring_out

        if(config['down_sample']):  
            self.lidar_in = downsample_lidar(self.lidar_out, config['up_factor'])
        else:
            self.lidar_in = copy.deepcopy(self.lidar_out)
            self.lidar_in['channels'] = int(config['res_in'].split('_')[0])
            self.lidar_in['points_per_ring'] = int(config['res_in'].split('_')[1])
        
        self.coord = generate_laser_directions(self.lidar_out)
        self.coord = torch.unsqueeze(torch.tensor(normalization_queries(self.coord, self.lidar_in)), 0)
        


        # backbone
        self.encoder = EDSRLiDAR(n_resblocks=16, n_feats=64, res_scale=1.0, use_xyz=config['use_xyz'])
        
        self.w_encoder = EDSRLiDAR(n_resblocks=4, n_feats=32, res_scale=1.0)
        self.w_predictor = nn.Sequential(ConvLiDAR(in_channels=32, out_channels=32, kernel_size=1,bias=True),
                                       nn.ReLU(),
                                       ConvLiDAR(in_channels=32, out_channels=1, kernel_size=1,bias=True),
                                       )

        self.inter = inter_double(num=4)

        

    def forward(self, batch_dict):
        
        batch_size = batch_dict['image_lr_batch'].shape[0]

        feat = self.encoder(batch_dict['image_lr_batch'])

        z, xy= self.inter(batch_dict['image_lr_batch'], feat, self.coord.repeat(batch_size, 1, 1))

        z = z.reshape(batch_size, 1, self.lidar_out['channels'], self.lidar_out['points_per_ring'])
        xy = xy.reshape(batch_size, 1, self.lidar_out['channels'], self.lidar_out['points_per_ring'])
        
        # z/xy to depth
        v_dir_sin = torch.unsqueeze(torch.sin(torch.linspace(start=self.lidar_out['min_v'], end=self.lidar_out['max_v'], steps=self.lidar_out['channels'])), dim=1)
        v_dir_sin = v_dir_sin.repeat(1, self.lidar_out['points_per_ring'])
        v_dir_sin = v_dir_sin.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        v_dir_cos = torch.unsqueeze(torch.cos(torch.linspace(start=self.lidar_out['min_v'], end=self.lidar_out['max_v'], steps=self.lidar_out['channels'])), dim=1)
        v_dir_cos = v_dir_cos.repeat(1, self.lidar_out['points_per_ring'])
        v_dir_cos = v_dir_cos.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        d_z = z / v_dir_sin.cuda()
        d_xy = xy / v_dir_cos.cuda()

        feat_z = self.w_encoder(d_z.detach())
        feat_xy = self.w_encoder(d_xy.detach())
        feat_dif = torch.abs(feat_z - feat_xy)
        w = self.w_predictor(feat_dif)
        w = torch.sigmoid(w)

        # Alway select HRV when v_dir_sin is to close to zero. VRV is the same.
        if not self.training:
            if self.dataset == 'Kitti':
                w[torch.abs(v_dir_sin) < 0.12] = 0
                w[torch.abs(v_dir_cos) < 0.12] = 1
            if self.dataset == 'Carla' and self.lidar_out['channels']==256:
                w[torch.abs(v_dir_sin) < 3e-2] = 0
                w[torch.abs(v_dir_cos) < 3e-2] = 1

        return d_z, d_xy, w


class WIN(pl.LightningModule):
    '''This model class is used to train all relevant models, including LiDAR-SR, TULIP, LIIF, ILN and our WIN.
    Input: config loaded from './configs/xxx.yaml' .
    One can train any method by simply modifile the 'model' field in config file.
    Since we use a different setting on KITTI dataset compared with TULIP, we retrained all the methods. For Calra dataset, we only have checkpoint of ILN and our WIN.
    See more details in our paper.
    '''
    def __init__(
        self,
        config
    ) -> None:

        super().__init__()

        self.config = config
        
        # output lidar
        self.lidar = initialize_lidar(os.path.join(config['data_dir'], 'lidar_specification.yaml'))
        self.lidar['channels'] = int(config['res_out'].split('_')[0])
        self.lidar['points_per_ring'] = int(config['res_out'].split('_')[1])

        # generate model
        self.model_name = config['model']
        self.net = generate_model(config['model'], config)

        # loss
        self.train_loss = MeanMetric()
        self.train_loss_z = MeanMetric()
        self.train_loss_xy = MeanMetric()
        self.train_loss_csm = MeanMetric()

        self.val_loss = MeanMetric()
        self.val_loss_z = MeanMetric()
        self.val_loss_xy = MeanMetric()
        self.val_loss_csm = MeanMetric()
        self.val_mae = MeanMetric()

        self.test_loss = MeanMetric()
        self.test_mae_z = MeanMetric()
        self.test_mae_xy = MeanMetric()
        self.test_mae = MeanMetric()
        self.test_best_mae = MeanMetric()
        self.test_IoU = MeanMetric()
        self.test_CD = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_mae_best = MinMetric()


    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins.
        We reset all the loss here.
        """
        self.train_loss.reset()
        self.train_loss_z.reset()
        self.train_loss_xy.reset()
        self.train_loss_csm.reset()

        self.val_loss.reset()
        self.val_loss_z.reset()
        self.val_loss_xy.reset()
        self.val_loss_csm.reset()
        self.val_mae_best.reset()
        self.val_mae.reset()

        self.test_mae_z.reset()
        self.test_mae_xy.reset()
        self.test_mae.reset()
        self.test_best_mae.reset()
        self.test_IoU.reset()
        self.test_CD.reset()
 
    def model_step(
        self, batch_dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        '''A single step of model during training or testing.
        Based on the input data, we calculate the reconstruction loss, selection loss, and MAE of the two branches respectively.
        '''

        # inference
        d_z, d_xy, gate = self.net(batch_dict)

        # loss for interpolation
        delta_z = torch.abs(d_z - batch_dict['range_image_hr_batch']) 
        delta_xy = torch.abs(d_xy - batch_dict['range_image_hr_batch'])
        mae_z = torch.mean(delta_z) * self.lidar['norm_r']
        mae_xy = torch.mean(delta_xy) * self.lidar['norm_r']
        loss_1 = mae_xy + mae_z 

        # loss for CSM
        normed_delta_z = delta_z.detach() / (batch_dict['range_image_hr_batch'] + 1e-10)
        normed_delta_xy = delta_xy.detach() / (batch_dict['range_image_hr_batch'] + 1e-10)
        normed_delta = torch.cat((normed_delta_z**2, normed_delta_xy**2), dim=1)
        theta = -2*(1.0 / 500)**2
        normed_delta = torch.softmax(normed_delta / theta, dim=1)
        gate_loss = torch.relu((normed_delta[:, [0], :, :] - gate) * torch.sign((normed_delta[:, [0], :, :] - 0.5)))
        loss_2 = torch.mean(gate_loss)

        # ablation of loss_2
        # target = torch.zeros_like(delta_z)
        # target[delta_z < delta_xy] = 1
        # loss_2_ = F.binary_cross_entropy(gate, target)
        # loss_2_ = torch.relu((target - gate) * torch.sign((target - 0.5)))
    
        return loss_1, loss_2, mae_z, mae_xy

    def training_step(
        self, batch, batch_idx: int
    ) -> torch.Tensor:
        '''A single step of training process.
        We log the total loss, loss of z branch, loss of d branch and loss of selection.
        '''
        loss_1, loss_2, mae_z, mae_xy = self.model_step(batch)

        # update and log training metrics
        self.train_loss(loss_1 + loss_2)
        self.train_loss_z(mae_z)
        self.train_loss_xy(mae_xy)
        self.train_loss_csm(loss_2)

        batch_size = self.config['batch_size']
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/loss_z", self.train_loss_z, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/loss_xy", self.train_loss_xy, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/loss_csm", self.train_loss_csm, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return loss_1 + loss_2

        
    def validation_step(self, batch_dict, batch_idx: int) -> None:
        '''A single step of validation.
        We obtain the final MAE that benefits from both views based on the confidence level predicted by the model.
        And val/mae is the monitor by which we save checkpoints.
        '''
        
        # inference
        d_z, d_xy, gate = self.net(batch_dict)

        # loss for interpolation
        delta_z = torch.abs(d_z - batch_dict['range_image_hr_batch']) 
        delta_xy = torch.abs(d_xy - batch_dict['range_image_hr_batch'])
        mae_z = torch.mean(delta_z) * self.lidar['norm_r']
        mae_xy = torch.mean(delta_xy) * self.lidar['norm_r']
        loss_1 = mae_xy + mae_z 

        # loss for CSM
        normed_delta_z = delta_z.detach() / (batch_dict['range_image_hr_batch'] + 1e-10)
        normed_delta_xy = delta_xy.detach() / (batch_dict['range_image_hr_batch'] + 1e-10)
        normed_delta = torch.cat((normed_delta_z**2, normed_delta_xy**2), dim=1)
        theta = -2*(1.0 / 500)**2
        normed_delta = torch.softmax(normed_delta / theta, dim=1)
        gate_loss = torch.relu((normed_delta[:, [0], :, :] - gate) * torch.sign((normed_delta[:, [0], :, :] - 0.5)))
        loss_2 = torch.mean(gate_loss)
        
        # choose the better one
        choose = gate > 0.5 * torch.ones_like(gate)
        delta = copy.copy(delta_xy.clone().detach())
        delta[choose] = delta_z[choose]
        mae = torch.mean(delta) * self.lidar['norm_r']

        # update and log validation metrics
        self.val_loss(loss_1 + loss_2)
        self.val_loss_z(mae_z)
        self.val_loss_xy(mae_xy)
        self.val_loss_csm(loss_2)
        self.val_mae(mae)

        batch_size = self.config['batch_size']
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/loss_z", self.val_loss_z, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/loss_xy", self.val_loss_xy, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/loss_gate", self.val_loss_csm, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)


    def on_validation_epoch_end(self) -> None:
        '''Update the best validation loss'''

        loss = self.val_mae.compute()
        self.val_mae_best(loss)
        self.log("val/mae_best", self.val_mae_best.compute(), sync_dist=True, prog_bar=True, batch_size=self.config['batch_size'])

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        
        # inference
        d_z, d_xy, gate = self.net(batch)
        delta_z = torch.abs(d_z - batch['range_image_hr_batch']) 
        delta_xy = torch.abs(d_xy - batch['range_image_hr_batch'])
        mae_z = torch.mean(delta_z) * self.lidar['norm_r']
        mae_xy = torch.mean(delta_xy) * self.lidar['norm_r']

        # mae
        choose = gate > 0.5 * torch.ones_like(gate)
        delta = copy.copy(delta_xy.clone().detach())
        delta[choose] = delta_z[choose]
        mae = torch.mean(delta) * self.lidar['norm_r']

        # best MAE refers to the best choice between HRV and VRV
        best_choose = delta_z <= delta_xy
        best_delta = copy.copy(delta_xy.clone().detach())
        best_delta[best_choose] = delta_z[best_choose]
        best_mae = torch.mean(best_delta) * self.lidar['norm_r']
        
        d = copy.copy(d_xy.clone().detach())
        d[choose] = d_z[choose]

        if self.config['dataset'] == 'KITTI':
            height, incl = get_kitti_param()
            pcd_pred = range_image_to_points_kitti(torch.squeeze(d), incl, height) * self.lidar['norm_r']
            pcd_gt = range_image_to_points_kitti(torch.squeeze(batch['range_image_hr_batch']), incl, height) * self.lidar['norm_r']
        else:
            pcd_pred = range_image_to_points(torch.squeeze(d), lidar=self.lidar, remove_zero_range=False) * self.lidar['norm_r']
            pcd_gt = range_image_to_points(torch.squeeze(batch['range_image_hr_batch']), lidar=self.lidar, remove_zero_range=False) * self.lidar['norm_r']


        # chamfer distance
        cd = get_cd(pcd_gt.unsqueeze(0).cuda(), pcd_pred.unsqueeze(0).cuda())

        # intersection of union(IoU)
        grid_size = 0.1
        pcd_all = np.vstack((pcd_pred.numpy(), pcd_gt.numpy()))
        min_coord = np.min(pcd_all, axis=0)
        max_coord = np.max(pcd_all, axis=0)
        voxel_grid_predicted = voxelize_point_cloud(pcd_pred.numpy(), grid_size, min_coord, max_coord)
        voxel_grid_ground_truth = voxelize_point_cloud(pcd_gt.numpy(), grid_size, min_coord, max_coord)
        iou, _, _ = get_iou(voxel_grid_predicted, voxel_grid_ground_truth)

        # update and log testing metrics
        self.test_mae_z(mae_z)
        self.test_mae_xy(mae_xy)
        self.test_mae(mae)
        self.test_best_mae(best_mae)
        self.test_IoU(iou)
        self.test_CD(cd)

        batch_size = self.config['batch_size']
        self.log("test/mae_z", self.test_mae_z, on_step=True, on_epoch=False, prog_bar=True, batch_size=batch_size)
        self.log("test/mae_xy", self.test_mae_xy, on_step=True, on_epoch=False, prog_bar=True, batch_size=batch_size)
        self.log("test/mae", self.test_mae, on_step=True, on_epoch=False, prog_bar=True, batch_size=batch_size)
        self.log("test/best_mae", self.test_best_mae, on_step=True, on_epoch=False, prog_bar=True, batch_size=batch_size)
        self.log("test/CD", self.test_CD, on_step=True, on_epoch=False, prog_bar=True, batch_size=batch_size)
        self.log("test/IoU", self.test_IoU, on_step=True, on_epoch=False, prog_bar=True, batch_size=batch_size)



    def on_test_epoch_end(self) -> None:
        batch_size = self.config['batch_size']
        self.log("test/mean_mae_z", self.test_mae_z.compute(), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("test/mean_mae_xy", self.test_mae_xy.compute(), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("test/mean_mae", self.test_mae.compute(), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("test/mean_beat_mae", self.test_best_mae.compute(), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("test/mean_CD", self.test_CD.compute(), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("test/mean_IoU", self.test_IoU.compute(), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        pass

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.net = self.net.train(True)
        else:
            self.net = self.net.train(False)

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        scheduler = MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250, 300, 350], gamma=0.5)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    

  
