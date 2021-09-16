class Config(object):
    epoch = 5000
    batch_size = 1024
    learning_rate = 0.001

    sigma = 1.0

    cuda = True
    gpu_cnt = 4

    async_loading = True
    pin_memory = True

    root_path = '/enter/root/path/'
    data_path = 'data/enter/remain/path'

    summary_dir = 'board'
    checkpoint_dir = 'trained'
    checkpoint_file = 'checkpoint.pth.tar'

    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2
    embedding_dim = 16