"""
Configuration file for the emotion detection pipeline.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Application Settings
APP_NAME = os.getenv('APP_NAME', 'emotion-ml')
APP_VERSION = os.getenv('APP_VERSION', '1.0.0')
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'notebooks')

# Data Paths
TRAIN_DATA_PATH = os.getenv('TRAIN_DATA_PATH', os.path.join(DATA_DIR, 'training_data.csv'))
TEST_DATA_PATH = os.getenv('TEST_DATA_PATH', os.path.join(DATA_DIR, 'test_data.csv'))

# Model Paths
EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, 'emotion_model.pkl')
INTENSITY_MODEL_PATH = os.path.join(MODELS_DIR, 'intensity_model.pkl')
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, 'preprocessor.pkl')

# Results Paths
PREDICTIONS_PATH = os.path.join(RESULTS_DIR, 'predictions.csv')
LOG_FILE = os.path.join(RESULTS_DIR, 'pipeline.log')

# Model Configuration
EMOTION_MODEL_TYPE = os.getenv('EMOTION_MODEL_TYPE', 'ensemble')
INTENSITY_MODEL_TYPE = os.getenv('INTENSITY_MODEL_TYPE', 'xgboost')
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.7'))

# Feature Engineering
MIN_TEXT_LENGTH = int(os.getenv('MIN_TEXT_LENGTH', '5'))
MAX_FEATURES = int(os.getenv('MAX_FEATURES', '100'))
USE_TF_IDF = os.getenv('USE_TF_IDF', 'True').lower() == 'true'
USE_SENTIMENT = os.getenv('USE_SENTIMENT', 'True').lower() == 'true'

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
VERBOSE = os.getenv('VERBOSE', 'True').lower() == 'true'

# Emotional States
EMOTIONAL_STATES = [
    'calm', 'anxious', 'content', 'restless', 'overwhelmed',
    'focused', 'neutral', 'sad', 'excited', 'frustrated', 'mixed'
]

# Intensity Range
MIN_INTENSITY = 1
MAX_INTENSITY = 5

# Time of Day Categories
TIME_OF_DAY_CATEGORIES = ['morning', 'afternoon', 'evening', 'night']

# Energy Level Categories
ENERGY_LEVELS = ['very_low', 'low', 'moderate', 'high', 'very_high']

# Stress Level Range
MIN_STRESS = 1
MAX_STRESS = 5

# Recommended Actions
RECOMMENDED_ACTIONS = [
    'deep_work', 'movement', 'journaling', 'grounding', 'box_breathing',
    'sound_therapy', 'yoga', 'pause', 'rest', 'light_planning'
]

# Timing Categories
TIMING_CATEGORIES = ['now', 'within_15_min', 'later_today', 'tonight', 'tomorrow_morning']

# Model Hyperparameters
RANDOM_FOREST_N_ESTIMATORS = 100
XGBOOST_N_ESTIMATORS = 100
XGBOOST_MAX_DEPTH = 6
LOGISTIC_REGRESSION_MAX_ITER = 1000
SVM_KERNEL = 'rbf'

# Random Seed for reproducibility
RANDOM_SEED = 42

# Display Settings
PANDAS_MAX_COLUMNS = None
PANDAS_MAX_ROWS = 100

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, NOTEBOOKS_DIR]:
    os.makedirs(directory, exist_ok=True)