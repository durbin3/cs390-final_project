class CONFIG:
    # project settings
    INPUT_SHAPE = (100, 100, 3)
    DOWN_SAMPLE_SCALE = 4  # dont change, model is hardcoded to use 4
    SAVE_DIR = 'saved_weights'
    HR_DIR = 'high_res'  # data directory
    BATCH_SIZE = 8  # make this as high your pc can handle
    SAVE_INTERVAL = 5
    LOG_INTERVAL = 500
    RESTART = False

    # training settings
    # each epoch is 8000 / BATCH_SIZE steps
    N_INIT_EPOCH = 200  # no. of initial generator training epoch
    N_EPOCH = 300  # no. of gan training epoch
    LR_START = 1e-4  # learning rate
    D_INPUT_RANDOM = 0.1  # amplitude of noise that is feed to the discriminator
    RAND_FLIP = 0.05  # percentage of data that has its label flipped
    UPDATE_D_EVERY = 50  # discriminator is update every ? steps

    # model settings
    B = 16  # number of residual blocks
    MOMENTUM = 0.9  # momentum of batchnorm
    VGG_WEIGHT = 1
    D_WEIGHT = 1e-3  # weight of discriminator loss in GAN
