
from model_utils import *

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset parameters
DATASET_ROOT = "./dataset"  # Path to store the dataset
PATH_ORIGINAL = "./path"  # Path to store the original model parameters
PATH_CRETRAIN = "./path_cretrain"  # Path to store the parameters after retraining
PATH_GAFN_GENERATOR = "./path_gafn_generator"  # Path to store the generator parameters of GAFN
PATH_UNLEARNING = "./path_unlearning"  # Path to store the parameters after unlearning
N_CLASSES = 10  # Number of classes in CIFAR10
IMG_SHAPE = (3, 32, 32)  # Shape of the image

# Training parameters
BATCH_SIZE = 32  # Number of samples per batch
NUM_EPOCHS = 50  # Total number of training epochs
SHUFFLE_DATA = True  # Whether to shuffle the data
NUM_WORKERS = 4  # Number of threads used for data loading
DROP_LAST = True  # Whether to drop the last batch if it can't be evenly divided

# Model parameters
LEARNING_RATE = 0.001  # Learning rate
BETA1 = 0.5  # Parameter for the Adam optimizer
NZ = 100  # Dimension of the noise vector
NC = 3  # Number of image channels
NGF = 128  # Number of generator feature maps
NDF = 128  # Number of discriminator feature maps

# Class selection parameters
FORGET_CLASSES = [0]  # Target classes to forget
SAMPLES_PER_CLASS = 5000  # Number of samples per class
ALL_CLASSES = list(range(N_CLASSES))  # All classes
RETAIN_CLASSES = list(set(ALL_CLASSES) - set(FORGET_CLASSES))  # Classes to retain

# Model class mapping
MODEL_CLASSES = {
    'AllCNN': AllCNN,
    'ResNet18': ResNet18,
    'ResNet50': ResNet50
}
