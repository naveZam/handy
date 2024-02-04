#conda install -c conda-forge keras
import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')
# Checking Version of Tensorflow

print("Version of Tensorflow: ", tf.__version__)

# Checking if cuda is there.

print("Cuda Availability: ", tf.test.is_built_with_cuda())

# Checking GPU is available or not.

print("GPU  Availability: ", tf.test.is_gpu_available())

# Check nos of GPUS

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


import tqdm
from tqdm import tqdm 

X = range(50000000)

sum = 0

for num in tqdm(X):
    sum = num + sum
    
print("Sum of numbers is : ", sum)