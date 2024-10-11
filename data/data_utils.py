import numpy as np 
import yaml
import copy
import torch

height = np.array(
      [0.20966667, 0.2092    , 0.2078    , 0.2078    , 0.2078    ,
       0.20733333, 0.20593333, 0.20546667, 0.20593333, 0.20546667,
       0.20453333, 0.205     , 0.2036    , 0.20406667, 0.2036    ,
       0.20313333, 0.20266667, 0.20266667, 0.20173333, 0.2008    ,
       0.2008    , 0.2008    , 0.20033333, 0.1994    , 0.20033333,
       0.19986667, 0.1994    , 0.1994    , 0.19893333, 0.19846667,
       0.19846667, 0.19846667, 0.12566667, 0.1252    , 0.1252    ,
       0.12473333, 0.12473333, 0.1238    , 0.12333333, 0.1238    ,
       0.12286667, 0.1224    , 0.12286667, 0.12146667, 0.12146667,
       0.121     , 0.12053333, 0.12053333, 0.12053333, 0.12006667,
       0.12006667, 0.1196    , 0.11913333, 0.11866667, 0.1182    ,
       0.1182    , 0.1182    , 0.11773333, 0.11726667, 0.11726667,
       0.1168    , 0.11633333, 0.11633333, 0.1154    ])

zenith = np.array([
        0.03373091,  0.02740409,  0.02276443,  0.01517224,  0.01004049,
        0.00308099, -0.00155868, -0.00788549, -0.01407172, -0.02103122,
       -0.02609267, -0.032068  , -0.03853542, -0.04451074, -0.05020488,
       -0.0565317 , -0.06180405, -0.06876355, -0.07361411, -0.08008152,
       -0.08577566, -0.09168069, -0.09793721, -0.10398284, -0.11052055,
       -0.11656618, -0.12219002, -0.12725147, -0.13407038, -0.14067839,
       -0.14510716, -0.15213696, -0.1575499 , -0.16711043, -0.17568678,
       -0.18278688, -0.19129293, -0.20247031, -0.21146846, -0.21934183,
       -0.22763699, -0.23536977, -0.24528179, -0.25477201, -0.26510582,
       -0.27326038, -0.28232882, -0.28893683, -0.30004392, -0.30953414,
       -0.31993824, -0.32816311, -0.33723155, -0.34447224, -0.352908  ,
       -0.36282001, -0.37216965, -0.38292524, -0.39164219, -0.39895318,
       -0.40703745, -0.41835542, -0.42777535, -0.43621111
    ]) 

incl = -zenith



def get_kitti_param():
    return height, incl

def range_image_to_points_kitti(range_image, incl, height):
    '''Back-projection of KITTI dataset.
    Due to the particularity of KITTI, we redefined the back-projection function to make the point cloud obtained by back-projection more accurate.
    '''
    points = np.zeros((64*1024,3))
    for i in range(64):
        line = range_image[i, :].detach().cpu().numpy() #[1024]
        phi = incl[63-i]
        h = height[63-i]
        k = np.sqrt(1 + np.tan(phi)**2)
        d_line = line / k
        col_inds = np.linspace(1024, 0, 1024, endpoint=False)
        azi_line = (col_inds) / 1024 * (2.0 * np.pi) - np.pi
        x_line = d_line * np.cos(azi_line)
        y_line = d_line * np.sin(azi_line)
        z_line = h - line * np.sin(phi) 
        points[i*1024:(i+1)*1024, 0] = x_line 
        points[i*1024:(i+1)*1024, 1] = y_line 
        points[i*1024:(i+1)*1024, 2] = z_line 
    
    distances = np.linalg.norm(points, axis=1)
    filtered_point_cloud = points[distances >= 1e-5]
    return torch.tensor(filtered_point_cloud, dtype=torch.float32)



def read_range_image_binary(file_path, dtype=np.float16, lidar=None):
    '''Read a range image from a binary file(.rimg).
    This function is used for Carla dataset.
    '''
    range_image_file = open(file_path, 'rb')

    # Read the size of range image
    size = np.fromfile(range_image_file, dtype=np.uint, count=2)
    
    # Read the range image
    range_image = np.fromfile(range_image_file, dtype=dtype)
    range_image = range_image.reshape(size[1], size[0])
    range_image = range_image.transpose()
    range_image = range_image.astype(np.float32)

    # Crop the values out of the detection range
    if lidar is not None:
        range_image[range_image < lidar['min_r']] = 0.0
        range_image[range_image > lidar['max_r']] = 0.0

    range_image_file.close()

    return torch.tensor(range_image.astype(np.float32))


def read_range_xyz_image_npy(file_path, dtype=np.float32):
    """Read a range image from a .npy file.
    This function is used for KITTI dataset.
    """
    range_intensity_image = np.load(file_path)
    range_image = range_intensity_image[..., :4]
    return torch.tensor(range_image.astype(dtype))

def initialize_lidar(file_path):
    """Initialize a LiDAR having given laser resolutions from a configuration file.
    """
    with open(file_path, 'r') as f:
        lidar = yaml.load(f, Loader=yaml.FullLoader)

    lidar['max_v'] *= (np.pi / 180.0)  # [rad]
    lidar['min_v'] *= (np.pi / 180.0)  # [rad]
    lidar['max_h'] *= (np.pi / 180.0)  # [rad]
    lidar['min_h'] *= (np.pi / 180.0)  # [rad]

    return lidar

def generate_laser_directions(lidar):
    '''Generate the laser directions using the LiDAR specification.
    It returns a set of the query laser directions, which is vital to implicit function methods.
    For correct search of neighborhood points, the lidar need to be set up carefully.
    '''

    v_dir = np.linspace(start=lidar['min_v'], stop=lidar['max_v'], num=lidar['channels'])
    h_dir = np.linspace(start=lidar['min_h'], stop=lidar['max_h'], num=lidar['points_per_ring'], endpoint=False)

    v_angles = []
    h_angles = []

    for i in range(lidar['channels']):
        v_angles = np.append(v_angles, np.ones(lidar['points_per_ring']) * v_dir[i])
        h_angles = np.append(h_angles, h_dir)

    return np.stack((v_angles, h_angles), axis=-1).astype(np.float32)

def range_image_to_points(range_image, lidar, remove_zero_range=True):
    """Convert a range image to the points in the sensor coordinate.
    This function works for all projections with fixed centers. Therefore we use it for CARLA dataset.
    """

    angles = generate_laser_directions(lidar)
    r = range_image.flatten().detach().cpu().numpy()
    x = np.sin(angles[:, 1]) * np.cos(angles[:, 0]) * r
    y = np.cos(angles[:, 1]) * np.cos(angles[:, 0]) * r
    z = np.sin(angles[:, 0]) * r

    points = np.stack((x, y, z), axis=-1)  # sensor coordinate

    # remove the points having invalid detection distances
    if remove_zero_range is True:
        points = np.delete(points, np.where(r < lidar['min_r']/lidar['norm_r']), axis=0)

    return torch.tensor(points)


def points_to_ranges(points):
    """Convert points in the sensor coordinate into the range data in spherical coordinate.
    """
    # sensor coordinate
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    r = np.sqrt(x * x + y * y + z * z)
    v = np.arctan2(z, np.sqrt(x * x + y * y))
    h = np.arctan2(x, y)
    return np.stack((v, h, r), axis=-1)


def points_to_image(points, lidar):
    '''Transform the points into a range-xyz image, where each pixel contains information in four dimensions, i.e. (r,x,y,z)
    '''
    range_samples = points_to_ranges(points)

    range_image = np.zeros([lidar['channels'], lidar['points_per_ring']], dtype=np.float32)
    xyz_image = np.zeros([3, lidar['channels'], lidar['points_per_ring']], dtype=np.float32)

    #max_y = max(lidar['max_v'], np.max(range_samples[:, 0]))
    #min_y = min(lidar['min_v'], np.min(range_samples[:, 0]))
    max_y = lidar['max_v']
    min_y = lidar['min_v']
    res_y = (max_y - min_y) / (lidar['channels']-1)  # include the last
    res_x = (lidar['max_h'] - lidar['min_h']) / lidar['points_per_ring']              # exclude the last

    # offset to match a point into a pixel center
    range_samples[:, 0] += (res_y * 0.5) # v
    range_samples[:, 1] += (res_x * 0.5) # h
    # horizontal values are within [-pi, pi)
    range_samples[range_samples[:, 1] < -np.pi, 1] += (2.0 * np.pi)
    range_samples[range_samples[:, 1] >= np.pi, 1] -= (2.0 * np.pi)
    # Pixel Index --> Ideally one pixel is assigned to one sample
    py = np.trunc((range_samples[:, 0] - lidar['min_v']) / res_y).astype(np.int16)
    px = np.trunc((range_samples[:, 1] - lidar['min_h']) / res_x).astype(np.int16)
    # filter the points out of the FOV
    index_filter = (py < lidar['channels']) & (py > -1)
    py = py[index_filter]
    px = px[index_filter]
    points = points[index_filter, :]
    range_samples = range_samples[index_filter, :]
    # Insert the ranges
    range_image[py, px] = range_samples[:, 2]
    xyz_image[:, py, px] = np.transpose(points)
    # Crop the values out of the detection range
    range_image[range_image < 10e-10] = 0

    # range_image[range_image < (lidar['min_r']/lidar['norm_r'])] = 0.0
    # range_image[range_image > (lidar['max_r']/lidar['norm_r'])] = 0.0
    # xyz_image[:, range_image < (lidar['min_r']/lidar['norm_r'])] = 0.0
    # xyz_image[:, range_image > (lidar['max_r']/lidar['norm_r'])] = 0.0

    return torch.cat([torch.unsqueeze(torch.tensor(range_image), 0), torch.tensor(xyz_image)], dim=0)




def downsample_range_image(range_im, downsample_rate):
    """Downsample the range image uniformly.
    To retain more useful information, we erase the row with the smallest vertical angle instead of the one with the largest vertical angle.
    """
    range_im_array = copy.copy(range_im).numpy()
    rows, cols = range_im.shape[:2]
    low_rows = int(rows / downsample_rate[0])
    low_cols = int(cols / downsample_rate[1])

    image = np.zeros((low_rows, low_cols), dtype=range_im_array.dtype)

    for i in range(low_rows):
        for j in range(low_cols):
            image[i, j] = range_im_array[(i+1) * downsample_rate[0] - 1, j * downsample_rate[1]]

    return torch.tensor(image)


def downsample_lidar(lidar, downsample_rate):
    """Downsample the lidar for the change of channels and min_v.
    This function is important for correct training of implicit function methods on real-word data, due to the fact that
    downsampling range image changes the minimum vertical angle, which determines how implicit function methods choose neighborhood.
    """
    lidar_lr = copy.copy(lidar)
    lidar_lr['channels'] = np.int16(lidar['channels'] / downsample_rate[0]) 
    v_res = (lidar['max_v'] - lidar['min_v']) / (lidar['channels'] - 1)
    lidar_lr['min_v'] = lidar['min_v'] + v_res * (downsample_rate[0] - 1) # change the min vertical angle

    lidar_lr['points_per_ring'] = np.int16(lidar['points_per_ring'] / downsample_rate[1])
    return lidar_lr

def normalization_queries(queries, lidar_in):
    """Normalize query lasers toward input range image space.
    [min_v-v_res*0.5 ~ max_v+v_res*0.5] --> [-1 ~ 1]
    [min_h-h_res*0.5 ~ max_h-h_res*0.5] --> [-1 ~ 1]
    """
    # Vertical angle: [min_v-v_res*0.5 ~ max_v+v_res*0.5] --> [0 ~ 1]
    v_res = (lidar_in['max_v'] - lidar_in['min_v']) / (lidar_in['channels'] - 1)
    min_v = lidar_in['min_v'] - v_res * 0.5
    max_v = lidar_in['max_v'] + v_res * 0.5
    queries[:, 0] -= min_v
    queries[:, 0] /= (max_v - min_v)

    # Horizontal angle: [min_h-h_res*0.5 ~ max_h-h_res*0.5] --> [0 ~ 1]
    h_res = (lidar_in['max_h'] - lidar_in['min_h']) / lidar_in['points_per_ring']
    queries[:, 1] += (h_res * 0.5)
    queries[queries[:, 1] < -np.pi, 1] += (2.0 * np.pi)  # min_h == -np.pi
    queries[queries[:, 1] >= np.pi, 1] -= (2.0 * np.pi)  # max_h == +np.pi
    queries[:, 1] += np.pi
    queries[:, 1] /= (2.0 * np.pi)

    # [0 ~ 1] --> [-1 ~ 1]
    queries *= 2.0
    queries -= 1.0
    return queries


def collate_fn_range_image(batch):
    '''This function is used by the dataloader to load batch data. 
    One can customize this function creates different inputs.
    '''
    batch_dict = dict()
    image_lr, range_image_hr=list(zip(*batch))

    batch_dict['image_lr_batch'] = torch.stack(image_lr)
    batch_dict['range_image_hr_batch'] = torch.stack(range_image_hr)

    return batch_dict
