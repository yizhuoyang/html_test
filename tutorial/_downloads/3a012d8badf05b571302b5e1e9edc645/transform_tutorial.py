"""
CSI Preprocessing.transform Tutorial
==============================================================
"""

######################################################################

# !pip install pysensing

######################################################################
# In this tutorial, we will be implementing a simple csi.preprocessing.transform tutorial using the pysensing library.
# 
import sys
sys.path.append('../..')
import pysensing.csi.preprocessing.transform as transform
import numpy as np

######################################################################
# remove_nan
# -----------------------------------

# remove_nan is a function that removes NaN values from the input data. It can replace NaN values with zero, mean, or median values.

test_data = [1, None, 3, np.inf, None, 6, -np.inf, 8, 9]
test_data1 = transform.remove_nan(test_data, interpolation_method='zero')
print(test_data1)
test_data2 = transform.remove_nan(test_data, interpolation_method='mean')
print(test_data2)
test_data3 = transform.remove_nan(test_data, interpolation_method='median')
print(test_data3)

######################################################################
# normalization
# ------------------------
# normalization is a function that normalizes the input data.

test_data4 = transform.normalization(test_data1)
print(test_data4)


######################################################################
# And that's it. We're done with our CSI augmentation.normalization tutorials. Thanks for reading.
