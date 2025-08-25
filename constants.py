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
SEGMENTED_DATASET = os.path.join(ROOT_DIR, 'data', '{}', 'segmented_dataset_{}')

# Model Paths (Disk)
MODEL_PATH = os.path.join(ROOT_DIR, 'disk', 'depth-save.pth')
#MODEL_PATH = os.path.join(ROOT_DIR, 'disk', 'epipolar-save.pth')

# Dataframe Paths
DATAFRAME_PATH = os.path.join(ROOT_DIR, 'data', '{}', 'processed_metadata.csv')

# Use Device
DEVICE = 'GPU'

# Descriptor Paths
SAVE_TRAIN_DESCRIPTORS_PATH = './data/{}/feature_descriptors_train_{}_{}/descriptors.h5'
SAVE_TEST_DESCRIPTORS_PATH = './data/{}/feature_descriptors_test_{}_{}/descriptors.h5'
SAVE_TRAIN_DESCRIPTORS_FOLDER = './data/{}/feature_descriptors_train_{}_{}/'
SAVE_TEST_DESCRIPTORS_FOLDER = './data/{}/feature_descriptors_test_{}_{}/'

# PCA and GMM Components
N_COMPONENTS_GMM = 256
N_COMPONENTS_PCA = 128 # Ovo promijenio zadnje da testiram.

# Pkl Paths
PCA_PATH = './data/{}/pca_model_{}_{}.pkl'
GMM_PATH = './data/{}/gmm_model_{}_{}.pkl'
FISHER_VECTORS = './data/{}/fisher_vectors_{}_{}.pkl'

# Database Path
DB_PATH = './data/{}/db/'

# Tmp Prediction Path
TMP = './data/tmp/'

# Evaluation results directory
EVALUATION_DIR = './evaluations/full_evals'
# XLSX file for saving population counting results
COUNT_RESULTS_XLSX = './evaluations/count/population_counting_results_ensamble_global_embedding.xlsx'
EVAL_RESULTS_XLSX = './evaluations/classification/classification_results_grid_search_final_new_NEW.xlsx'

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
GEOMETRIC_CANDIDATES = 10

# Geometric Verification Method Selection
GV_METHOD = "RANSAC"  # Options: "RANSAC" or "MAGSAC"

# Geometric Verification Scaling Constants
MAX_INLIERS_FOR_SCALING = 20      # Cap inliers at this value for exponential formula
LOG_SCALING_FACTOR = 1.0          # Scaling factor for logarithmic approach
LINEAR_COMBINATION_ALPHA = 0.7    # Weight for Fisher distance in linear combination
SIGMOID_STEEPNESS = 0.1           # Controls sigmoid curve steepness
SIGMOID_MIDPOINT = 20             # Midpoint for sigmoid scaling
MAX_REASONABLE_INLIERS = 50       # Expected maximum inliers for normalization

# For multi-scaling
MULTISCALE_SCALES = [0.75, 1.0, 1.25]
MAX_FEATURES_PER_SCALE = 2000  # Limit features to control memory usage
ENABLE_MULTISCALE = False  # Toggle for easy comparison with single-scale


MAX_GMM_DESCRIPTORS = 2000000
# Cap descriptors contributed by each image when stacking for PCA/GMM training.
MAX_DESCRIPTORS_PER_IMAGE = 5000

# Tradeoff between geomtric verification and fisher similarity
ALPHA = 0.35

#Max number of extracted keypoints
MAX_KEYPOINTS = 5000

# Default weights for combining Fisher vectors from multiple feature
# extraction methods when using the ``ensamble`` option.
ENSEMBLE_WEIGHTS = [1/3, 1/3, 1/3]
