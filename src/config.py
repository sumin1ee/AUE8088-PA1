import os

# Training Hyperparameters
NUM_CLASSES         = 200
BATCH_SIZE          = 2048
VAL_EVERY_N_EPOCH   = 1

NUM_EPOCHS          = 50

# Optimize Configs
OPTIMIZER_PARAMS = {'type': 'SGD', 'lr': 0.01, 'momentum': 0.9}
# OPTIMIZER_PARAMS = {'type': 'RMSprop', 'lr': 0.01, 'alpha': 0.99}
# OPTIMIZER_PARAMS    = {'type': 'AdamW', 'lr': 0.01, 'weight_decay': 1e-4}
# OPTIMIZER_PARAMS = {'type': 'Adam', 'lr': 0.01, 'betas': (0.9, 0.999)}
# OPTIMIZER_PARAMS = {'type': 'Adagrad', 'lr': 0.01}
# OPTIMIZER_PARAMS = {'type': 'Adadelta', 'lr': 0.01}

# Scheduler Config4
# SCHEDULER_PARAMS    = {'type': 'MultiStepLR', 'milestones': [30, 35], 'gamma': 0.2}
SCHEDULER_PARAMS    = {'type': 'CosineAnnealingWarmRestarts', 'T_0': 10, 'T_mult': 4, 'eta_min': 1e-5}

# Dataset
DATASET_ROOT_PATH   = os.path.join(os.getcwd(), 'src/datasets')
NUM_WORKERS         = 8

# Augmentation
IMAGE_ROTATION      = 20
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 64
IMAGE_PAD_CROPS     = 4
IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
IMAGE_STD           = [0.2302, 0.2265, 0.2262]
IMAGE_RESIZE_FOR_VIT = 224

# Network
MODEL_NAME          = 'resnet50'

# Compute related
ACCELERATOR         = 'gpu'
DEVICES             = [0]
PRECISION_STR       = '32-true'

# Logging
WANDB_PROJECT       = 'aue8088-pa1-comparing-models'
WANDB_ENTITY        = os.environ.get('WANDB_ENTITY')
print(f'WANDB_ENTITY: {WANDB_ENTITY}')
WANDB_SAVE_DIR      = 'wandb/'
WANDB_IMG_LOG_FREQ  = 50
WANDB_NAME          = f'{MODEL_NAME}-{OPTIMIZER_PARAMS["type"]}'
WANDB_NAME         += f'-{OPTIMIZER_PARAMS["lr"]:.2f}'
# WANDB_NAME         += f'-{SCHEDULER_PARAMS["type"]}{OPTIMIZER_PARAMS["lr"]:.1E}'
