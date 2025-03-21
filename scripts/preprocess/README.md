# 2D & 3D Data Preprocessing

## Overview

This document provides instructions for pre-processing both 3D and 2D data for different datasets, including 
- ScanNet


One can also use the provided code as a reference for pre-processing your customized dataset.

## Prerequisites

### Environment
Before you begin, simply activate the `openscene` conda environment.

Alternatively, make sure the following package installed:
- `torch`
- `numpy`
- `plyfile`
- `opencv-python`
- `imageio`
- `pandas`
- `tqdm`

### Download the original dataset
- **ScanNet**: Download ScanNet v2 data from the [official ScanNet website](https://github.com/ScanNet/ScanNet).


For preprocessing 3D point clouds with GT labels, one can simply run:
```bash
python preprocess_3d_scannet.py
```

For preprocessing 2D RGB-D images, one can also simply run:
```bash
python preprocess_2d_scannet.py
```

**Note**: In the code, you might need to modify the following:
- `in_path`: path to the original downloaded dataset
- `out_dir`: output directory to save your processed data
- `scene_list`: path to the list containing all scenes
- `split`: choose from `train`/`val`/`test` data to process



## Folder structure
Once running the pre-processing code above, you should have a data structure like below. Here we choose the processed ScanNet as an example:

```
data/
│
├── scannet_2d
│   │
│   ├── scene0000_00
│   │   ├── color
│   │   ├── depth
│   │   ├── pose
│   │   └── intrinsic 
│   │
│   ├── scene0000_01
│   │   ├── color
│   │   ├── depth
│   │   ├── pose
│   │   └── intrinsic
│   │
│   └── ...
|   |
|   └── intrinsics.txt (fixed intrinsic parameters for all scenes)
│
└── scannet_3d
    │
    ├── train
    │   ├── scene0000_00.pth
    │   ├── scene0000_01.pth
    │   ├── ...
    │   ├── scene0706_00.pth
    │
    └── val
        ├── scene0011_00.pth
        └── ...
    
```

**Customized dataset**: Make sure that you have the same structure after preprocessing.
