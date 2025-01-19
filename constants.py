# Segmentation Model
MODEL_NAME = 'isnet-genera-use'

# Segmented Dataset Path
SEGMENTED_DATASET = './data/segmented_dataset/'

# Model Paths (Disk)
MODEL_PATH = './disk/depth-save.pth'
#MODEL_PATH = './disk/epipolar-save.pth'

# Dataframe Paths
DATAFRAME_PATH = './data/processed_metadata.csv'

# Use Device
DEVICE = 'GPU'

# Descriptor Paths
SAVE_TRAIN_DESCRIPTORS_PATH = './data/feature_descriptors_train/descriptors.h5'
SAVE_TEST_DESCRIPTORS_PATH = './data/feature_descriptors_test/descriptors.h5'
SAVE_TRAIN_DESCRIPTORS_FOLDER = './data/feature_descriptors_train/'
SAVE_TEST_DESCRIPTORS_FOLDER = './data/feature_descriptors_test/'

# PCA and GMM Components
N_COMPONENTS_GMM = 2
N_COMPONENTS_PCA = 123

# Pkl Paths
PCA_PATH = './data/pca_model.pkl'
GMM_PATH = './data/gmm_model.pkl'
FISHER_VECTORS = './data/fisher_vectors.pkl'

# Database Path
DB_PATH = './data/db/'

# Tmp Prediction Path
TMP = './data/tmp/'