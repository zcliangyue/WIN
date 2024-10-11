import torch
import numpy as np
from data.data_utils import *
from chamfer3D.dist_chamfer_3D import chamfer_3DDist

def get_cd(pc1, pc2):
    '''Get the chamfer distace of two points clouds.
    The input and output are both torch.tensor.
    '''
    chd = chamfer_3DDist()
    dist1, dist2, _, _ = chd(pc1, pc2)
    CD_dist = torch.mean(dist1, dim=1, keepdims=True) + torch.mean(dist2, dim=1, keepdims=True)
    return torch.mean(CD_dist)

def voxelize_point_cloud(point_cloud, grid_size, min_coord, max_coord):
    '''Turn the point-based representation into voxel-based.
    The retured grid is used to calculate IoU by the function 'get_iou'
    '''
    # Calculate the dimensions of the voxel grid
    dimensions = ((max_coord - min_coord) / grid_size).astype(int) + 1

    # Create the voxel grid
    voxel_grid = np.zeros(dimensions, dtype=bool)

    # Assign points to voxels
    indices = ((point_cloud - min_coord) / grid_size).astype(int)
    voxel_grid[tuple(indices.T)] = True

    return voxel_grid

def get_iou(voxel_grid_predicted, voxel_grid_ground_truth):
    '''Get the IoU metric of two voxels.
    We use the same code as TULIP.
    '''
    intersection = np.logical_and(voxel_grid_predicted, voxel_grid_ground_truth)
    union = np.logical_or(voxel_grid_predicted, voxel_grid_ground_truth)

    iou = np.sum(intersection) / np.sum(union)

    true_positive = np.sum(intersection)
    false_positive = np.sum(voxel_grid_predicted) - true_positive
    false_negative = np.sum(voxel_grid_ground_truth) - true_positive

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    return iou, precision, recall
