class Config(object):
    data_name = 'cifar10'
    
    epoch = 500
    batch_size = 1024
    learning_rate = 0.001

    sigma = 1.0

    cuda = True
    gpu_cnt = 1

    async_loading = True
    pin_memory = True

    root_path = '/home/M2016551/DeepHashing/'
    data_directory = 'data'
    summary_dir = 'board'
    checkpoint_dir = 'trained'
    checkpoint_file = 'checkpoint.pth.tar'

    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2
    embedding_dim = 16