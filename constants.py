import os
# Change this, due to recent changes in code, this is largely deprecated.

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Segmentation Model
ISNET_MODEL_NAME = 'isnet-general-use'
SEGMENTATION_MODEL_TYPE = 'sam'
SAM_ISNET_IOU_THETA = 0.3  # IoU threshold for SAM with ISNet

SAM_MODEL_TYPE = 'vit_h'  # Options: vit_b, vit_l, vit_h
SAM_CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'models', 'sam_vit_h_4b8939.pth')  # Path to the SAM model checkpoint

SAM2_MODEL_NAME = 'sam2.1_hiera_small.pt'             
SAM2_CFG  = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CHECKPOINT_PATH = os.path.join(ROOT_DIR, 'models', SAM2_MODEL_NAME)

# Segmented Dataset Path
SEGMENTED_DATASET = os.path.join(ROOT_DIR, 'data', '{}', 'segmented_dataset_unsegmented')

# Model Paths (Disk)
MODEL_PATH = os.path.join(ROOT_DIR, 'disk', 'depth-save.pth')
#MODEL_PATH = os.path.join(ROOT_DIR, 'disk', 'epipolar-save.pth')

# Dataframe Paths
DATAFRAME_PATH = os.path.join(ROOT_DIR, 'data', '{}', 'processed_metadata.csv')

# Use Device
DEVICE = 'GPU'

# Descriptor Paths
SAVE_TRAIN_DESCRIPTORS_PATH = './data/{}/feature_descriptors_train/descriptors.h5'
SAVE_TEST_DESCRIPTORS_PATH = './data/{}/feature_descriptors_test/descriptors.h5'
SAVE_TRAIN_DESCRIPTORS_FOLDER = './data/{}/feature_descriptors_train/'
SAVE_TEST_DESCRIPTORS_FOLDER = './data/{}/feature_descriptors_test/'

# PCA and GMM Components
N_COMPONENTS_GMM = 256 # Preko 128 ne ide!
N_COMPONENTS_PCA = 96 # Ovo promijenio zadnje da testiram.

# Pkl Paths
PCA_PATH = './data/{}/pca_model_{}.pkl'
GMM_PATH = './data/{}/gmm_model_{}.pkl'
FISHER_VECTORS = './data/{}/fisher_vectors_{}.pkl'

# Database Path
DB_PATH = './data/{}/db/'

# Tmp Prediction Path
TMP = './data/tmp/'

# Evaluation results directory
EVALUATION_DIR = './evaluations/'
# XLSX file for saving population counting results
COUNT_RESULTS_XLSX = './evaluations/count/population_counting_results.xlsx'

WILD_DATASET_PATH = './data/wildlifedatasets/wildlifereid-10k/versions/7'


# Keynet, Hardnet, Affnet
PATCH_SIZE = 32

# Geometric Verification Constants
RATIO_THRESHOLD = 0.8           # Lowe's ratio test threshold for feature matching
INLIER_THRESHOLD = 0.5        # RANSAC inlier threshold for animal re-identification
MIN_MATCHES = 4                 # Minimum number of matches required for RANSAC
MIN_INLIERS = 10                 # Minimum number of inliers for reliable geometric verification
INSUFFICIENT_MATCHES_PENALTY = 10.0  # Penalty multiplier for insufficient matches
POOR_GEOMETRY_PENALTY = 5.0     # Penalty multiplier for poor geometric consistency
FISHER_DISTANCE_MIN_CLAMP = 0.01     # Minimum clamp value for Fisher distance
FISHER_DISTANCE_MAX_CLAMP = 1.0      # Maximum clamp value for Fisher distance
NORMALIZED_THRESHOLD_DIVISOR = 100.0 # Divisor for normalizing RANSAC threshold

# Geometric Verification Scaling Constants
MAX_INLIERS_FOR_SCALING = 20      # Cap inliers at this value for exponential formula
LOG_SCALING_FACTOR = 1.0          # Scaling factor for logarithmic approach
LINEAR_COMBINATION_ALPHA = 0.7    # Weight for Fisher distance in linear combination
SIGMOID_STEEPNESS = 0.1           # Controls sigmoid curve steepness
SIGMOID_MIDPOINT = 20             # Midpoint for sigmoid scaling
MAX_REASONABLE_INLIERS = 50       # Expected maximum inliers for normalization

# For multi-scaling
MULTISCALE_SCALES = [0.5, 1.0, 1.5]
MAX_FEATURES_PER_SCALE = 1500  # Limit features to control memory usage
ENABLE_MULTISCALE = False  # Toggle for easy comparison with single-scale


MAX_GMM_DESCRIPTORS = 1600000