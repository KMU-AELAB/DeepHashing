import os
import shutil
import random
from tqdm import tqdm

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from graph.model.vae import VAE as Model
from graph.loss.sample_loss import Loss
from data.dataset import Dataset_CIFAR10, Dataset_CIFAR100

from utils.metrics import AverageMeter, mAP
from utils.train_utils import set_logger, count_model_prameters

from tensorboardX import SummaryWriter

cudnn.benchmark = True


class Sample(object):
    def __init__(self, config):
        self.config = config
        self.batch_size = self.config.batch_size
        self.best_map = 0.

        self.logger = set_logger('train_epoch.log')

        # define dataloader
        if 'cifar' in self.config.data_name:
            self.train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                transforms.RandomErasing(p=0.6, scale=(0.03, 0.08), ratio=(0.3, 3.3)),
            ])

            if self.config.data_name == 'cifar10':
                self.train_dataset = Dataset_CIFAR10(os.path.join(self.config.root_path, self.config.data_directory),
                                                     train=True, download=True, transform=self.train_transform)
                self.test_dataset = Dataset_CIFAR10(os.path.join(self.config.root_path, self.config.data_directory),
                                                    train=False, download=True, transform=self.train_transform)
            elif self.config.data_name == 'cifar100':
                self.train_dataset = Dataset_CIFAR100(os.path.join(self.config.root_path, self.config.data_directory),
                                                      train=True, download=True, transform=self.train_transform)
                self.test_dataset = Dataset_CIFAR100(os.path.join(self.config.root_path, self.config.data_directory),
                                                     train=False, download=True, transform=self.train_transform)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=2,
                                           pin_memory=self.config.pin_memory)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2,
                                          pin_memory=self.config.pin_memory)

        # define models
        self.model = Model(self.config.num_hiddens, self.config.num_residual_layers,
                           self.config.num_residual_hiddens, self.config.embedding_dim).cuda()

        # define loss
        self.loss = Loss().cuda()

        # define lr
        self.lr = self.config.learning_rate

        # define optimizer
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # define optimize scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', factor=0.8, cooldown=8)

        # initialize train counter
        self.epoch = 0

        self.manual_seed = random.randint(10000, 99999)

        torch.manual_seed(self.manual_seed)
        torch.cuda.manual_seed_all(self.manual_seed)
        random.seed(self.manual_seed)

        # parallel setting
        gpu_list = list(range(self.config.gpu_cnt))
        self.model = nn.DataParallel(self.model, device_ids=gpu_list)

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

        # Summary Writer
        self.summary_writer = SummaryWriter(log_dir=os.path.join(self.config.root_path, self.config.summary_dir),
                                            comment='BarGen')
        self.print_train_info()

    def print_train_info(self):
        print("seed: ", self.manual_seed)
        print('Number of model parameters: {}'.format(count_model_prameters(self.model)))

    def load_checkpoint(self, file_name):
        filename = os.path.join(self.config.root_path, self.config.checkpoint_dir, file_name)
        try:
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.opt.load_state_dict(checkpoint['optimizer'])

        except OSError as e:
            print("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            print("**First time to train**")

    def save_checkpoint(self):
        tmp_name = os.path.join(self.config.root_path, self.config.checkpoint_dir,
                                'checkpoint_{}.pth.tar'.format(self.epoch))

        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer': self.opt.state_dict(),
        }

        torch.save(state, tmp_name)
        shutil.copyfile(tmp_name, os.path.join(self.config.root_path, self.config.checkpoint_dir,
                                               self.config.checkpoint_file))

    def run(self):
        try:
            self.train()

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        for _ in range(self.config.epoch):
            self.epoch += 1
            self.train_by_epoch()

            if not self.epoch % 10:
                self.test()
        self.test()

    def train_by_epoch(self):
        tqdm_batch = tqdm(self.train_dataloader, total=len(self.train_dataloader), leave=False,
                          desc="epoch-{}".format(self.epoch))

        avg_loss = AverageMeter()
        self.model.train()
        for curr_it, data in enumerate(tqdm_batch):
            self.opt.zero_grad()

            origin = data['origin'].cuda(async=self.config.async_loading)
            trans = data['trans'].cuda(async=self.config.async_loading)

            origin_recon, origin_feature, origin_code = self.model(origin)
            trans_recon, trans_feature, trans_code = self.model(trans)

            loss = self.loss(origin_code, trans_code, origin_feature, trans_feature)
            loss.backward()
            self.opt.step()

            avg_loss.update(loss)

        tqdm_batch.close()

        self.scheduler.step(avg_loss.val)

    def test(self):
        train_code, test_code, train_label, test_label = [], [], [], []

        self.model.eval()
        with torch.no_grad():
            # train dataset
            tqdm_batch = tqdm(self.train_dataloader, total=len(self.train_dataloader), leave=False)
            for curr_it, data in enumerate(tqdm_batch):
                origin = data['origin'].cuda(async=self.config.async_loading)
                target = data['target'].cuda(async=self.config.async_loading)

                _, _, code = self.model(origin)

                train_code.append(code)
                train_label.append(target)

            tqdm_batch.close()
            train_code, train_label = torch.cat(train_code), torch.cat(train_label)

            # test dataset
            tqdm_batch = tqdm(self.test_dataloader, total=len(self.train_dataloader), leave=False)
            for curr_it, data in enumerate(tqdm_batch):
                origin = data['origin'].cuda(async=self.config.async_loading)
                target = data['target'].cuda(async=self.config.async_loading)

                _, _, code = self.model(origin)

                test_code.append(code)
                test_label.append(target)

            tqdm_batch.close()
            test_code, test_label = torch.cat(test_code), torch.cat(test_label)

            map = mAP(train_code, test_code, train_label, test_label)

        if map > self.best_map:
            self.best_map = map
            self.save_checkpoint()

        print(f'--- epoch{self.epoch} mAP: {map} / best mAP: {self.best_map} ---')
