"""
Tooth Alignment Data Preparation
================================

"""


######################################################################
# Description
# ~~~~~~~~~~~
# 
# A target pose transformation prediction network for each tooth based on
# their pointcloud. ### Environment Setup \* System: Ubuntu 20.04 \*
# Language: Python 3.9 \* Deep Learning Framework: Pytorch 2.4.0, CUDA
# 12.5 ### Workflow \* Original Data: A separate single-tooth model in the
# local coordinate system, original pose description (tooth number +
# quaternion rotation + translation), target pose description. \*
# Preprocessing: Filter data and perform farthest point sampling on the
# tooth model to obtain the sampled pointcloud. \* Input: Tooth pointcloud
# in global coordinate system (centered) + 28 \* 3D original tooth center
# positions. \* Output: 28 \* 7D target pose transformations (quaternion
# rotation + translation).
# 


######################################################################
# Raw Data Format
# ~~~~~~~~~~~~~~~
# 


######################################################################
# Preprocessing
# ~~~~~~~~~~~~~
# 
# Filter data and perform farthest point sampling to obtain the sampled
# (global coordinate) pointcloud, and save the original and target
# transformations to the data folder.
# 


######################################################################
# Tooth Segmentation Data Preparation
# ===================================
# 


######################################################################
# Description
# ~~~~~~~~~~~
# 
# A target FDI label prediction network for each point in dentition based
# on its pointcloud. ### Environment Setup \* System: Ubuntu 20.04 \*
# Language: Python 3.9 \* Deep Learning Framework: Pytorch 2.4.0, CUDA
# 12.5 ### Workflow \* Original Data: 3D intra-oral scans for upper or
# lower jaws, FDI labels for each vertices. Download from
# https://osf.io/xctdy. \* Preprocessing: Transform the .obj to pointcloud
# and extract features for each point. \* Input: Point features and its
# jaw’s category ([1, 0] for lower and [0, 1] for upper). \* Output: FDI
# labels for each point.
# 


######################################################################
# Raw Data Format
# ~~~~~~~~~~~~~~~
# 


######################################################################
# Preprocessing
# ~~~~~~~~~~~~~
# 
# Transform the original IOS objects to pointclouds and extract features
# (3D Cartesian Coordinates, Normal Vector, Gaussian Curvature, Average
# Angle Curvature) for each point. Finally save the features, categories
# and labels to the data folder.
# 