class CONFIG:
    B = 16
    DOWN_SAMPLE_SCALE = 4
    N_INIT_EPOCH = 100
    N_EPOCH = 1000
    INPUT_SHAPE = (256, 256, 3)
    HR_DIR = 'high_res'
    LR_START = 1e-4
    BATCH_SIZE = 2
    SAVE_INTERVAL = 100
    SAVE_DIR = 'saved_weights'
