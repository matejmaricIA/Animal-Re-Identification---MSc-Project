import os
# Change this, due to recent changes in code, this is largely deprecated.

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Segmentation (Grounded SAM2)
# Place GroundingDINO config + checkpoint and SAM2 checkpoint under ./models/.
GROUNDING_DINO_CONFIG_PATH = os.path.join(
    ROOT_DIR, "models", "GroundingDINO_SwinT_OGC.py"
)
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(
    ROOT_DIR, "models", "groundingdino_swint_ogc.pth"
)
SAM2_CHECKPOINT_PATH = os.path.join(ROOT_DIR, "models", "sam2.1_hiera_large.pt")
# Config path is resolved relative to the installed sam2 package if not absolute.
SAM2_CONFIG_REL_PATH = "configs/sam2.1/sam2.1_hiera_l.yaml"

# GroundingDINO thresholds
DINO_BOX_THRESHOLD = 0.3
DINO_TEXT_THRESHOLD = 0.25

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
COUNT_RESULTS_XLSX = './evaluations/count/population_counting_results_UPDATED.xlsx'
EVAL_RESULTS_XLSX = './evaluations/classification/classification_results_LATE_FUSION_tier3_GV_CAL.xlsx'

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
GEOMETRIC_CANDIDATES = 200
UNION_CANDIDATES = 600
LOCAL_RANK_CANDIDATES = 200

# Geometric Verification Method Selection
GV_METHOD = "MAGSAC"  # Options: "RANSAC" or "MAGSAC"

# Geometric Verification Scaling Constants
MAX_INLIERS_FOR_SCALING = 20      # Cap inliers at this value for exponential formula
LOG_SCALING_FACTOR = 1.0          # Scaling factor for logarithmic approach
LINEAR_COMBINATION_ALPHA = 0.7    # Weight for Fisher distance in linear combination
SIGMOID_STEEPNESS = 0.1           # Controls sigmoid curve steepness
SIGMOID_MIDPOINT = 20             # Midpoint for sigmoid scaling
MAX_REASONABLE_INLIERS = 50       # Expected maximum inliers for normalization



MAX_GMM_DESCRIPTORS = 4000000
# Cap descriptors contributed by each image when stacking for PCA/GMM training.
MAX_DESCRIPTORS_PER_IMAGE = 2500

# Tradeoff between geomtric verification and fisher similarity
ALPHA = 0.35

#Max number of extracted keypoints
MAX_KEYPOINTS = 2500

# Default weights for combining Fisher vectors from multiple feature
# extraction methods when using the ``ensamble`` option.
ENSEMBLE_WEIGHTS = [1/3, 1/3, 1/3]

# MegaDescriptor training/split metadata derived from MegaDescriptor_l_384_baseline.csv.
# Keys are lowercase dataset names for case-insensitive lookup.
MD_DATASET_SPLITS = {
    "aerialcattle2017": {"split_type": "random", "trained_on": True, "random_split": True},
    "amvrakikosturtles": {"split_type": "similarity-aware", "trained_on": False, "random_split": False},
    "atrw": {"split_type": "random", "trained_on": True, "random_split": True},
    "belugaid": {"split_type": "random", "trained_on": True, "random_split": True},
    "catindividualimages": {"split_type": "similarity-aware", "trained_on": False, "random_split": False},
    "chicks4freeid": {"split_type": "similarity-aware", "trained_on": False, "random_split": False},
    "cowdataset": {"split_type": "time-aware", "trained_on": False, "random_split": False},
    "ctai": {"split_type": "random", "trained_on": True, "random_split": True},
    "czoo": {"split_type": "random", "trained_on": True, "random_split": True},
    "dogfacenet": {"split_type": "similarity-aware", "trained_on": False, "random_split": False},
    "elpephants": {"split_type": "original_split", "trained_on": False, "random_split": False},
    "friesiancattle2015": {"split_type": "random", "trained_on": True, "random_split": True},
    "friesiancattle2017": {"split_type": "random", "trained_on": True, "random_split": True},
    "giraffes": {"split_type": "random", "trained_on": True, "random_split": True},
    "giraffezebraid": {"split_type": "random", "trained_on": True, "random_split": True},
    "hyenaid2022": {"split_type": "random", "trained_on": True, "random_split": True},
    "ipanda50": {"split_type": "random", "trained_on": True, "random_split": True},
    "leopardid2022": {"split_type": "random", "trained_on": True, "random_split": True},
    "liondata": {"split_type": "random", "trained_on": True, "random_split": True},
    "macaquefaces": {"split_type": "random", "trained_on": True, "random_split": True},
    "mpdd": {"split_type": "similarity-aware", "trained_on": False, "random_split": False},
    "multicamcows2024": {"split_type": "time-aware", "trained_on": False, "random_split": False},
    "nyaladata": {"split_type": "random", "trained_on": True, "random_split": True},
    "opencows2020": {"split_type": "random", "trained_on": True, "random_split": True},
    "polarbearvidid": {"split_type": "similarity-aware", "trained_on": False, "random_split": False},
    "primface": {"split_type": "similarity-aware", "trained_on": False, "random_split": False},
    "reunionturtles": {"split_type": "similarity-aware", "trained_on": False, "random_split": False},
    "seastarreid2023": {"split_type": "time-aware", "trained_on": False, "random_split": False},
    "seaturtleid2022": {"split_type": "time-aware", "trained_on": False, "random_split": False},
    "sealid": {"split_type": "random", "trained_on": True, "random_split": True},
    "southernprovinceturtles": {"split_type": "similarity-aware", "trained_on": False, "random_split": False},
    "stripespotter": {"split_type": "random", "trained_on": True, "random_split": True},
    "whalesharkid": {"split_type": "random", "trained_on": True, "random_split": True},
    "zakynthosturtles": {"split_type": "similarity-aware", "trained_on": False, "random_split": False},
    "zinditurtlerecall": {"split_type": "random", "trained_on": True, "random_split": True},
}
