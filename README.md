<div align="center">   

# WIN: Variable-View Implicit LIDAR Upsampling Network
</div>

# News
- **[2025/3/15]** 🚀 Repository Initialization.

# Abstract

LiDAR upsampling aims to increase the resolution of sparse point sets obtained from low-cost sensors, providing better performance for various downstream tasks (e.g., Autonomous Driving, High Definitation Map). Existing methods transfer LiDAR point cloud into range view, and focus on designing complex encoders or interpolation strategies to improve the resolution of LiDAR range images. However, our analysis shows that using the range view inevitably results in the loss of geometric information. We propose a Variable-View Implicit LiDAR Upsampling network, named **WIN** to solve this problem. It decouples range views into two novel virtual view representations, **Horizon Range View (HRV)** and **Vertical Range View (VRV)**. The key idea behind this is that introducing more perspectives can make up for the geometric information lost in a single perspective. We also prove theoretically that the proposed virtual view representation has a smaller error range compared to the range view representation. In addition, we design two novel strategies (i.e., contrast selection module and selection loss) to fuse the upsampling results of these two virtual representations and stabilize the whole training process. As a result, compared with the current state-of-the art (SOTA) method ILN, WIN introduces only 0.4M additional parameters, yet achieves a **+4.53%** increase in the MAE and a **+7.01%** increase in the IoU on the CARLA dataset. Furthermore, our method also outperforms all existing methods in downstream tasks (i.e., Depth Completion and Localization). The pre-trained model and code will be released upon acceptance.

<img src="figures\effect.png" alt="effect" width="50%" />

# Overall Framework

<img src="figures\framework.png" alt="overall freamwork" style="zoom:50%;" />

# Install

```cmd
conda create -n win python=3.8
conda activate win
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch-lightning==1.6.0
pip install -r requirments.txt
```

# Datasets

Our experiments include both synthetic and real-world datasets:

- **Synthetic datasets**: we use a virtual dataset built with [CARLA](https://carla.org/) simulator, following the settings of TULIP and ILN. The synthetic data consists of noise-free point clouds with a vertical FoV of 30$^\circ$ and three resolutions: \(64\times 1024\), \(128\times 2048\), and \(256\times 4096\). This dataset can be downloaded from the [link](https://sgvr.kaist.ac.kr/~yskwon/papers/icra22-iln/carla.zip).
- **Real-world dataset**: We use the KITTI dataset, collected with a Velodyne HDL-64E LiDAR, featuring a vertical FoV of 26.8$^\circ$. To generate ground truth range images, we transformed the raw point clouds into range images with a resolution of \(64\times 1024\). For testing, we sampled frames uniformly from sequences of \textit{2011\_10\_03}, while the remained sequences are used for training. This result in a (18,336/2,804) train/test split. 

# Training and Testing

Training and testing on CARLA dataset:

```python
python train.py --config configs/train_carla.yaml
python test.py --config configs/test_carla.yaml
```

Training and testing on KITTI dataset:

```
python train.py --config configs/train_kitti.yaml
python test.py --config configs/test_kitti.yaml
```

## Checkpoints

We will release the checkpoints soon.
