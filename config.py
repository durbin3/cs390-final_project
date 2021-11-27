class CONFIG:
    B = 16
    DOWN_SAMPLE_SCALE = 4
    N_INIT_EPOCH = 30
    N_EPOCH = 1000
    INPUT_SHAPE = (256, 256, 3)
    HR_DIR = 'high_res'
    LR_START = 1e-4
    BATCH_SIZE = 2
    BATCH_SIZE_INIT = 8
    BATCH_SIZE_D = 8
    SAVE_INTERVAL = 1
    SAVE_INTERVAL_INIT = 10
    PREVIEW_INTERVAL = 1
    ALTERNATE_INTERVAL = 10
    USE_INIT = False
    SAVE_DIR = 'saved_weights'
    LOAD_WEIGHTS = False
    VGG_WEIGHT = 1
    D_WEIGHT = 1e-3
    MSE_WEIGHT = 1e-4
    D_INPUT_RANDOM = 0.1
    RAND_FLIP = 0.05
    MOMENTUM = 0.5
