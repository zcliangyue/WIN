import torch
import torch.nn as nn
import torch.nn.functional as F

from model.components.edsr_lidar import EDSRLiDAR
from model.components.tf_weight import WeightTransformer, MLP
from einops import rearrange


def make_coord(shape, ranges=None, flatten=True):
    # Make coordinates at grid centers.
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class ILN(nn.Module):
    '''The implement of ILN.
    '''
    def __init__(self, d=1, h=8, use_xyz=False):
        super().__init__()
        self.d = d  # depth
        self.h = h  # num of heads

        # Encoder: EDSR
        self.encoder = EDSRLiDAR(n_resblocks=16, n_feats=64, res_scale=1.0, use_xyz=use_xyz)

        # Attention: ViT
        dim = self.encoder.out_dim
        self.attention = WeightTransformer(num_classes=1, dim=dim,
                                           depth=self.d, heads=self.h, mlp_dim=dim,
                                           dim_head=(dim//self.h), dropout=0.1)

    def gen_feat(self, inp):
        self.inp_img = inp
        self.feat = self.encoder(inp)
        return self.feat

    def query_detection(self, coord):
        feat = self.feat
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda().permute(2, 0, 1).unsqueeze(0).expand(
            feat.shape[0], 2, *feat.shape[-2:])

        # visualize
        feat_coord_vis = torch.squeeze(rearrange(feat_coord, "t n q c -> (q c) n t"))

        # N: batch size
        # Q: query size
        # D: feature dimension
        # T: neighbors
        preds = torch.empty((4, coord.shape[0], coord.shape[1]), device='cuda')
        rel_coords = torch.empty((4, coord.shape[0], coord.shape[1], coord.shape[2]), device='cuda')
        q_feats = torch.empty((4, feat.shape[0], coord.shape[1], feat.shape[1]), device='cuda')
        #
        q_coords = torch.empty((4, coord.shape[0], coord.shape[1], coord.shape[2]), device='cuda')

        n, q = coord.shape[:2]
        t = 0
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift  # vertical
                coord_[:, :, 1] += vy * ry + eps_shift  # horizontal
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # q_feat: z_t           [N, Q, D]
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1).cuda(), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                # q_coord: q_t          [N, Q, 2]
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1).cuda(), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                # rel_coord: del q_t    [N, Q, 2]
                rel_coord = coord.cuda() - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                # pred: r_t             [N, Q, 1]
                pred = F.grid_sample(self.inp_img[: ,[0] ,: ,:], coord_.flip(-1).unsqueeze(1).cuda(), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                preds[t, :, :] = pred.view(n, q)    # [4, N, Q]
                rel_coords[t, :, :, :] = rel_coord  # [4, N, Q, 2]
                q_feats[t, :, :, :] = q_feat        # [4, N, Q, D]
                #
                q_coords[t, :, :, :] = q_coord

                t = t + 1

        q_feats = rearrange(q_feats, "t n q d -> (n q) t d")
        rel_coords = rearrange(rel_coords, "t n q c -> (n q) t c")
        #
        q_coords = rearrange(q_coords, "t n q c -> (n q) t c")

        weights = self.attention(q_feats, rel_coords)               # [N*Q, 4, 1]
        preds = rearrange(preds, "t n q -> (n q) t").unsqueeze(1)   # [N*Q, 1, 4]

        ret = torch.matmul(preds, weights)  # [N*Q, 1]
        ret = ret.view(n, q, -1)            # [N, Q, 1]

        return ret

    def forward(self, inp, coord):
        self.gen_feat(inp)
        return self.query_detection(coord)
    


class ILN_Interpolate(nn.Module):
    """This module is a interpolation module seperated from ILN. 
    We optimized the code in the original ILN code regarding neighborhood sampling, resulting in a significant increase in training efficiency. 
    Specific differences can be checked by examining this class and the ILN class.
    """
    def __init__(self, d=1, h=8, dim=64):
        super().__init__()
        self.d = d  # depth
        self.h = h  # num of heads
        
        self.encoder_z = MLP(in_dim=4*dim, out_dim=4, hidden_list=[4*dim, 2*dim, dim])
        # Attention: ViT
        self.attention = WeightTransformer(num_classes=1, dim=dim,
                                           depth=self.d, heads=self.h, mlp_dim=dim,
                                           dim_head=(dim//self.h), dropout=0.1)


    def query_detection(self, coord):
        feat = self.feat
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda().permute(2, 0, 1).unsqueeze(0).expand(
            feat.shape[0], 2, *feat.shape[-2:])

        # N: batch size
        # Q: query size
        # D: feature dimension
        # T: neighbors
        preds = torch.empty((4, coord.shape[0], coord.shape[1]), device='cuda')
        rel_coords = torch.empty((4, coord.shape[0], coord.shape[1], coord.shape[2]), device='cuda')
         
        q_feats = torch.empty((4, feat.shape[0], coord.shape[1], feat.shape[1]), device='cuda')
        #
        q_coords = torch.empty((4, coord.shape[0], coord.shape[1], coord.shape[2]), device='cuda')

        n, q = coord.shape[:2]
        t = 0
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift  # vertical
                coord_[:, :, 1] += vy * ry + eps_shift  # horizontal
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                coord_ = coord_.flip(-1).unsqueeze(1)

                # q_feat: z_t           [N, Q, D]
                q_feat = F.grid_sample(feat, coord_, mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                # q_coord: q_t          [N, Q, 2]
                q_coord = F.grid_sample(feat_coord, coord_, mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                # # pred: r_t             [N, Q, 1]
                pred = F.grid_sample(self.depth[:, [0], :, :], coord_, mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                preds[t, :, :] = torch.squeeze(pred)    # [4, N, Q]

                rel_coords[t, :, :, :] = rel_coord  # [4, N, Q, 2]
                q_feats[t, :, :, :] = q_feat        # [4, N, Q, D]
               
                t = t + 1

        q_feats = rearrange(q_feats, "t n q d -> (n q) t d")
        rel_coords = rearrange(rel_coords, "t n q c -> (n q) t c")

        q_coords = rearrange(q_coords, "t n q c -> (n q) t c")

        weights = self.attention(q_feats, rel_coords)               # [N*Q, 4, 1]

        preds = rearrange(preds, "t n q -> (n q) t").unsqueeze(1)   # [N*Q, 1, 4]

        ret = torch.matmul(preds, weights)  # [N*Q, 1]
        ret = ret.view(n, q, -1)            # [N, Q, 1]

        
        return ret

    def forward(self, depth, feat, coord):
        # input is feat
        self.depth = depth
        self.feat = feat

        self.d = depth[: ,[0] ,: ,:]
        

        return self.query_detection(coord)
    

class interpolate_xy(nn.Module):
    """This module is a HRV interpolation module. 
    The only difference between this class and ILN is that the sampling process is performed on the horizontal range view.
    """
    def __init__(self, d=1, h=8, dim=64):
        super().__init__()
        self.d = d  # depth
        self.h = h  # num of heads
        self.num=4
        num = self.num
        self.encoder_xy = MLP(in_dim=num*dim, out_dim=num, hidden_list=[num*dim, num//2*dim, dim])
        self.pos_embedd = MLP(in_dim=2*num, out_dim=num*dim, hidden_list=[dim, num//2*dim])
        # Attention: ViT
        self.attention = WeightTransformer(num_classes=1, dim=dim,
                                           depth=self.d, heads=self.h, mlp_dim=dim,
                                           dim_head=(dim//self.h), dropout=0.1)


    def query_detection(self, coord):
        feat = self.feat
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda().permute(2, 0, 1).unsqueeze(0).expand(
            feat.shape[0], 2, *feat.shape[-2:])

        # N: batch size
        # Q: query size
        # D: feature dimension
        # T: neighbors
        preds_xy = torch.empty((self.num, coord.shape[0], coord.shape[1]), device='cuda')

        rel_coords = torch.empty((self.num, coord.shape[0], coord.shape[1], coord.shape[2]), device='cuda')
         
        q_feats = torch.empty((self.num, feat.shape[0], coord.shape[1], feat.shape[1]), device='cuda')
        

        n, q = coord.shape[:2]
        t = 0
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift  # vertical
                coord_[:, :, 1] += vy * ry + eps_shift  # horizontal
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # q_feat: z_t           [N, Q, D]
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1).cuda(), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                # q_coord: q_t          [N, Q, 2]
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1).cuda(), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                # rel_coord: del q_t    [N, Q, 2]
                rel_coord = coord.cuda() - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                # pred: r_t             [N, Q, 1]
                pred_xy = F.grid_sample(self.xy, coord_.flip(-1).unsqueeze(1).cuda(), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)


                preds_xy[t, :, :] = pred_xy.view(n, q)    # [4, N, Q]

                rel_coords[t, :, :, :] = rel_coord  # [4, N, Q, 2]
                q_feats[t, :, :, :] = q_feat        # [4, N, Q, D]
               
                t = t + 1

        q_feats = rearrange(q_feats, "t n q d -> (n q) (t d)")

        rel_coords = rearrange(rel_coords, "t n q c -> (n q) (t c)")
    
        q_feats = q_feats + self.pos_embedd(rel_coords)
        # [N*Q, 4, 1]      
        weights_xy = torch.softmax(self.encoder_xy(q_feats), dim=1).unsqueeze(2)       
        #weights_xy = self.attention(q_feats, rel_coords)
        # [N*Q, 1, 4]
        preds_xy = rearrange(preds_xy, "t n q -> (n q) t").unsqueeze(1)
        # [N*Q, 1]
        ret_xy = torch.matmul(preds_xy, weights_xy) 

        # [N, Q, 1]          
        ret_xy = ret_xy.view(n, q, -1)                  



        return ret_xy

    def forward(self, depth, feat, coord):
        # input is feat
        self.depth = depth
        self.feat = feat

        x = depth[: ,[1] ,: ,:]
        y = depth[: ,[2] ,: ,:]
        self.xy = torch.sqrt(x**2 + y**2)
        

        return self.query_detection(coord)
        
class interpolate_z(nn.Module):
    """This module is a VRV interpolation module. 
    The only difference between this class and ILN is that the sampling process is performed on the vertical range view.
    """
    def __init__(self, d=1, h=8, dim=64):
        super().__init__()
        self.d = d  # depth
        self.h = h  # num of heads
        
        self.encoder_z = MLP(in_dim=4*dim, out_dim=4, hidden_list=[4*dim, 2*dim, dim])
        # Attention: ViT
        self.attention = WeightTransformer(num_classes=1, dim=dim,
                                           depth=self.d, heads=self.h, mlp_dim=dim,
                                           dim_head=(dim//self.h), dropout=0.1)


    def query_detection(self, coord):
        feat = self.feat
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda().permute(2, 0, 1).unsqueeze(0).expand(
            feat.shape[0], 2, *feat.shape[-2:])

        # N: batch size
        # Q: query size
        # D: feature dimension
        # T: neighbors
        preds = torch.empty((4, coord.shape[0], coord.shape[1]), device='cuda')
        rel_coords = torch.empty((4, coord.shape[0], coord.shape[1], coord.shape[2]), device='cuda')
         
        q_feats = torch.empty((4, feat.shape[0], coord.shape[1], feat.shape[1]), device='cuda')
        #
        q_coords = torch.empty((4, coord.shape[0], coord.shape[1], coord.shape[2]), device='cuda')

        n, q = coord.shape[:2]
        t = 0
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift  # vertical
                coord_[:, :, 1] += vy * ry + eps_shift  # horizontal
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                # q_feat: z_t           [N, Q, D]
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1).cuda(), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                # q_coord: q_t          [N, Q, 2]
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1).cuda(), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                # rel_coord: del q_t    [N, Q, 2]
                rel_coord = coord.cuda() - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                # pred: r_t             [N, Q, 1]
                pred = F.grid_sample(self.z, coord_.flip(-1).unsqueeze(1).cuda(), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                preds[t, :, :] = pred.view(n, q)    # [4, N, Q]
                rel_coords[t, :, :, :] = rel_coord  # [4, N, Q, 2]
                q_feats[t, :, :, :] = q_feat        # [4, N, Q, D]
                #
                q_coords[t, :, :, :] = q_coord

                t = t + 1

        q_feats = rearrange(q_feats, "t n q d -> (n q) t d")
        rel_coords = rearrange(rel_coords, "t n q c -> (n q) t c")
        #
        q_coords = rearrange(q_coords, "t n q c -> (n q) t c")

        weights = self.attention(q_feats, rel_coords)               # [N*Q, 4, 1]

        preds = rearrange(preds, "t n q -> (n q) t").unsqueeze(1)   # [N*Q, 1, 4]

        ret = torch.matmul(preds, weights)  # [N*Q, 1]
        ret = ret.view(n, q, -1)            # [N, Q, 1]

        
        return ret

    def forward(self, depth, feat, coord):
        # input is feat
        self.depth = depth
        self.feat = feat

        self.z = depth[: ,[3] ,: ,:]
        

        return self.query_detection(coord)


class inter_double(nn.Module):
    """This module is for both HRV and VRV interpolation. 
    It performs interpolation in two view representations respectively, and predict a confidence score by 'gatenet', which is a simple MLP.
    """
    def __init__(self, d=1, h=8, dim=64, num=4):
        super().__init__()
        self.d = d  # depth
        self.h = h  # num of heads
        self.dim = dim
        self.num = num

        self.encoder_z = MLP(in_dim=num*dim, out_dim=num, hidden_list=[num*dim, num//2*dim, dim])
        self.encoder_xy = MLP(in_dim=num*dim, out_dim=num, hidden_list=[num*dim, num//2*dim, dim])
        self.pos_embedd = MLP(in_dim=2*num, out_dim=num*dim, hidden_list=[dim, num//2*dim])
        self.gatenet = MLP(in_dim=num*dim, out_dim=1, hidden_list=[4*dim, 2*dim, dim])

        self.encoder_z.initialize()
        self.encoder_xy.initialize()
        self.pos_embedd.initialize()
        self.gatenet.initialize()

    def query_detection(self, xy, z, feat, coord):
        
        vx_lst = []
        for i in range(self.num // 2):
            vx_lst.append(1- self.num // 2 + i * 2)

        vy_lst = [-1, 1]
        eps_shift = 1e-6

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        feat_coord = make_coord(feat.shape[-2:], flatten=False).permute(2, 0, 1).unsqueeze(0).expand(
            feat.shape[0], 2, *feat.shape[-2:]).cuda()

        # N: batch size
        # Q: query size
        # D: feature dimension
        # T: neighbors
        preds_z = torch.empty((self.num, coord.shape[0], coord.shape[1]), device='cuda') 
        preds_xy = torch.empty((self.num, coord.shape[0], coord.shape[1]), device='cuda')
        rel_coords = torch.empty((self.num, coord.shape[0], coord.shape[1], coord.shape[2]), device='cuda')
        q_feats = torch.empty((self.num, feat.shape[0], coord.shape[1], feat.shape[1]), device='cuda')
        
        n, q = coord.shape[:2]
        t = 0
        
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift  # vertical
                coord_[:, :, 1] += vy * ry + eps_shift  # horizontal
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                coord_ = coord_.flip(-1).unsqueeze(1)

                # q_feat: z_t           [N, Q, D]
                q_feat = F.grid_sample(feat, coord_, mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                # q_coord: q_t          [N, Q, 2]
                q_coord = F.grid_sample(feat_coord, coord_, mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                # pred: r_t             [N, Q, 1]
                pred_z = F.grid_sample(z, coord_, mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
                pred_xy = F.grid_sample(xy, coord_, mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]

                preds_z[t, :, :] = torch.squeeze(pred_z)    # [4, N, Q]
                preds_xy[t, :, :] = torch.squeeze(pred_xy)   # [4, N, Q]

                rel_coords[t, :, :, :] = rel_coord  # [4, N, Q, 2]
                q_feats[t, :, :, :] = q_feat        # [4, N, Q, D]
               
                t = t + 1

        q_feats = rearrange(q_feats, "t n q d -> (n q) (t d)")
        rel_coords = rearrange(rel_coords, "t n q c -> (n q) (t c)")
        
        q_feats = q_feats + self.pos_embedd(rel_coords)

        
        weights_z = torch.softmax(self.encoder_z(q_feats), dim=1).unsqueeze(2)   # [N*Q, 4, 1]       
        weights_xy = torch.softmax(self.encoder_xy(q_feats), dim=1).unsqueeze(2) # [N*Q, 4, 1]        

        preds_z = rearrange(preds_z, "t n q -> (n q) t").unsqueeze(1)   # [N*Q, 1, 4]
        preds_xy = rearrange(preds_xy, "t n q -> (n q) t").unsqueeze(1) # [N*Q, 1, 4]
        
        ret_z = torch.matmul(preds_z, weights_z)  
        ret_xy = torch.matmul(preds_xy, weights_xy) 

        ret_z = ret_z.view(n, q, -1)            
        ret_xy = ret_xy.view(n, q, -1)              

        return ret_z, ret_xy

    def forward(self, depth, feat, coord):
        # input: features of range image
        self.depth = depth
        self.feat = feat

        x = depth[:, [1], :, :]
        y = depth[:, [2], :, :]
        xy = torch.sqrt(x**2 + y**2)
        z = depth[: ,[3], :, :]

        return self.query_detection(xy, z, feat, coord.cuda())