# Dataset size
# N_TRAIN_SAMPLES = 309871
N_VAL_SAMPLES = 154937
N_TEST_SAMPLES = 116203
NUM_FEATURES = 54
NUM_CLASSES = 7

# All feature columns in the data
LABEL_COLUMN = "Covertype"

BOOL_COLUMNS = [
    "Wilderness_Area1",
    "Wilderness_Area2",
    "Wilderness_Area3",
    "Wilderness_Area4",
    "Soil_Type1",
    "Soil_Type2",
    "Soil_Type3",
    "Soil_Type4",
    "Soil_Type5",
    "Soil_Type6",
    "Soil_Type7",
    "Soil_Type8",
    "Soil_Type9",
    "Soil_Type10",
    "Soil_Type11",
    "Soil_Type12",
    "Soil_Type13",
    "Soil_Type14",
    "Soil_Type15",
    "Soil_Type16",
    "Soil_Type17",
    "Soil_Type18",
    "Soil_Type19",
    "Soil_Type20",
    "Soil_Type21",
    "Soil_Type22",
    "Soil_Type23",
    "Soil_Type24",
    "Soil_Type25",
    "Soil_Type26",
    "Soil_Type27",
    "Soil_Type28",
    "Soil_Type29",
    "Soil_Type30",
    "Soil_Type31",
    "Soil_Type32",
    "Soil_Type33",
    "Soil_Type34",
    "Soil_Type35",
    "Soil_Type36",
    "Soil_Type37",
    "Soil_Type38",
    "Soil_Type39",
    "Soil_Type40",
]

INT_COLUMNS = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]

STR_COLUMNS = []
STR_NUNIQUESS = []

FLOAT_COLUMNS = []

ENCODED_CATEGORICAL_COLUMNS = []

# Model hyperparameters
FEATURE_DIM = 64
OUTPUT_DIM = 64
NUM_DECISION_STEPS = 5
RELAXATION_FACTOR = 1.5
BATCH_MOMENTUM = 0.7
VIRTUAL_BATCH_SIZE = 512

# Training parameters
TRAIN_FILE = "data/train_covertype.csv"
VAL_FILE = "data/val_covertype.csv"
TEST_FILE = "data/test_covertype.csv"
MAX_STEPS = 1000000
DISPLAY_STEP = 1000
VAL_STEP = 10000
SAVE_STEP = 40000
TEST_STEP = 1000
INIT_LEARNING_RATE = 0.02
DECAY_EVERY = 500
DECAY_RATE = 0.95
BATCH_SIZE = 16384
SPARSITY_LOSS_WEIGHT = 0.0001
GRADIENT_THRESH = 2000.0
SEED = 1
REDUCED = True
MODEL_NAME = "tabnet_forest_covertype_reduced_model" if REDUCED else "tabnet_forest_covertype_model"
