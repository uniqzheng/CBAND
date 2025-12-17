from argparse import ArgumentParser
import os
import h5py
import torch
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
from scipy import stats
from tensorboardX import SummaryWriter
import datetime
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from tqdm import tqdm
import sys

import warnings

warnings.filterwarnings("ignore")

class Logger(object):
    def __init__(self, fileN='Default.log'):
        self.terminal = sys.stdout
        sys.stdout = self
        self.log = open(fileN, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def reset(self):
        self.log.close()
        sys.stdout = self.terminal

    def flush(self):
        pass

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    # 4-parameter logistic function
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat


class VQADataset(Dataset):
    def __init__(self, features_dir='CNN_features_KoNViD-1k/', index=None, max_len=240, feat_dim=2048, scale=1, is_train=True):
        super(VQADataset, self).__init__()
        self.features = []
        self.length = np.zeros((len(index), 1))
        self.mos = np.zeros((len(index), 1))
        for i in range(len(index)):
            features = np.load(features_dir + str(index[i]) + '_resnet50_stage2_ggd2.npy')
            self.length[i] = features.shape[0]
            self.features.append(features)
            self.mos[i] = np.load(features_dir + str(index[i]) + '_score.npy')  
        self.scale = scale 
        self.label = self.mos / self.scale  # label normalization
        self.is_train = is_train
        self.features = np.array(self.features, dtype=object)

    def __len__(self):
        return len(self.mos)

    def __getitem__(self, idx):
        if self.is_train:
            frame_idx = np.sort(np.random.choice(int(self.length[idx]), replace=False, size=1))
            frame_idx = int(frame_idx[0])
            sample = self.features[idx][frame_idx], self.length[idx], self.label[idx]
        else:
            sample = self.features[idx], self.length[idx], self.label[idx]
        return sample

class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset: VQADataset, num_repeats: int=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx: int):
        return self.dataset[idx % self.dataset.__len__()]

class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

class fc_layers(nn.Module):
    def __init__(self, ):
        super(fc_layers, self).__init__()
        self.additional_fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128,1)
        )

    def forward(self, input):
        score = self.additional_fc(input)
        return score

# for vgg16feat_stage2
# class fc_layers(nn.Module):
#     def __init__(self, ):
#         super(fc_layers, self).__init__()
#         self.additional_fc = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.Dropout(0.2),
#             nn.ReLU(),
#             nn.Linear(128,64),
#             nn.Dropout(0.2),
#             nn.ReLU(),
#             nn.Linear(64,32),
#             nn.Dropout(0.2),
#             nn.ReLU(),
#             nn.Linear(32,1)
#         )

#     def forward(self, input):
#         score = self.additional_fc(input)
#         return score


if __name__ == "__main__":
    parser = ArgumentParser(description='"Training features')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='learning rate (default: 0.00001)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--datarepeat', type=int, default=25,
                        help='number of dataset repeat (default: 25)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--database', default='LIVE_Banding', type=str,
                        help='database name (default: LIVE_Banding)')
    parser.add_argument('--model', default='Resnet50', type=str,
                        help='model name (default: VSFA)')
    parser.add_argument('--exp_id', default=0, type=int,
                        help='exp id for train-val-test splits (default: 0)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='test ratio (default: 0.2)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='val ratio (default: 0.2)')

    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='weight decay (default: 0.0)')

    parser.add_argument("--notest_during_training", action='store_true',
                        help='flag whether to test during training')
    parser.add_argument("--disable_visualization", action='store_true',
                        help='flag whether to enable TensorBoard visualization')
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args()

    args.decay_interval = int(args.epochs / 10)
    args.decay_ratio = 0.8

    logger = Logger('/workspace/CBAND/logs/log_LIVE_Banding_resnet50_stage2_ggd2_%s.txt' % args.exp_id)

    torch.manual_seed(args.seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.database == 'KoNViD-1k':
        features_dir = '/workspace/CBAND/features/CNN_features_KoNViD-1k_Resnet50_stage2_ggd2/'  # features dir
        datainfo = '/workspace/CBAND/data/KoNViD-1kinfo.mat'  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
    if args.database == 'LIVE_Banding':
        features_dir = '/workspace/CBAND/features/CNN_features_LIVE_Banding_Resnet50_stage2_ggd2/'
        datainfo = '/workspace/CBAND/data/LIVE_Banding_contentaware.mat'

    print('EXP ID: {}'.format(args.exp_id))
    print(args.database)
    print(args.model)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    Info = h5py.File(datainfo, 'r')  # index, ref_ids
    index = Info['index']
    index = index[:, args.exp_id % index.shape[1]]  # np.random.permutation(N)
    ref_ids = Info['ref_ids'][0, :]  #
    max_len = int(Info['max_len'][0])
    trainindex = index[0:int(np.ceil((1 - args.test_ratio - args.val_ratio) * len(index)))]
    testindex = index[int(np.ceil((1 - args.test_ratio) * len(index))):len(index)]
    train_index, val_index, test_index = [], [], []
    for i in range(len(ref_ids)):
        train_index.append(i) if (ref_ids[i] in trainindex) else \
            test_index.append(i) if (ref_ids[i] in testindex) else \
                val_index.append(i)

    scale = Info['scores'][0, :].max()  # label normalization factor
    train_dataset = VQADataset(features_dir, train_index, max_len, scale=scale, is_train=True)
    train_dataset = DatasetRepeater(train_dataset, args.datarepeat)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = VQADataset(features_dir, val_index, max_len, scale=scale, is_train=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset)
    test_dataset = VQADataset(features_dir, test_index, max_len, scale=scale, is_train=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset)

    print('length of train_dataloader: ', len(train_loader))
    print('length of val_dataloader: ', len(val_loader))
    print('length of test_dataloader: ', len(test_loader))

    model = fc_layers().to(device)  #
    # model = CustomDataParallel(model)

    if not os.path.exists('models'):
        os.makedirs('models')
    trained_model_file = 'models/{}-{}-EXP{}'.format(args.model, args.database, args.exp_id)
    if not os.path.exists('results'):
        os.makedirs('results')
    save_result_file = 'results/{}-{}-EXP{}'.format(args.model, args.database, args.exp_id)

    if not args.disable_visualization:  # Tensorboard Visualization
        writer = SummaryWriter(log_dir='{}/EXP{}-{}-{}-{}-{}-{}-{}'
                               .format(args.log_dir, args.exp_id, args.database, args.model,
                                       args.lr, args.batch_size, args.epochs,
                                       datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))

    criterion = nn.L1Loss()  # L1 loss
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)
    best_val_criterion = -1  # SROCC min
    with tqdm(total=args.epochs) as pbar:
        for epoch in range(args.epochs):
            # Train
            model.train()
            L = 0

            for i, (features, length, label) in enumerate(train_loader):
                features = features.to(device).float()
                label = label.to(device).float()
                optimizer.zero_grad()  #
                outputs = model(features)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                L = L + loss.item()
            train_loss = L / (i + 1)
            pbar.update()
            pbar.set_postfix(**{'train loss (epoch)': train_loss})

            model.eval()
            
            # Val
            if args.val_ratio > 0:
                y_pred_val = np.zeros(len(val_index))
                y_val = np.zeros(len(val_index))
                L = 0

                with torch.no_grad():
                    for i, (features, length, label) in enumerate(val_loader):
                        # label is normalized in dataset, recover to original scale for metrics
                        y_val[i] = scale * label.item()

                        features = features.to(device).float()
                        features = torch.squeeze(features, dim=0)  # keep same behavior as your original code
                        label = label.to(device).float()

                        outputs = model(features)

                        # your original code uses mean(outputs)
                        y_pred_val[i] = scale * (torch.mean(outputs).item())

                        loss = criterion(outputs, label)
                        L += loss.item()

                val_loss = L / (i + 1)
                val_SROCC = stats.spearmanr(y_pred_val, y_val)[0]
                val_KROCC = stats.kendalltau(y_pred_val, y_val)[0]

                # logistic regression between y_pred & y
                beta_init = [np.max(y_val), np.min(y_val), np.mean(y_pred_val), 0.5]
                popt, _ = curve_fit(logistic_func, y_pred_val, y_val, p0=beta_init, maxfev=int(1e8))
                y_pred_val_logistic = logistic_func(y_pred_val, *popt)

                val_PLCC = stats.pearsonr(y_val, y_pred_val_logistic)[0]
                val_RMSE = np.sqrt(mean_squared_error(y_val, y_pred_val_logistic))
            else:
                # if no val split, don't update best
                val_SROCC = -np.inf

            # Update the model with the best val_SROCC
            if args.val_ratio > 0 and (val_SROCC > best_val_criterion):
                print("Val results: val loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
                    .format(val_loss, val_SROCC, val_KROCC, val_PLCC, val_RMSE))
                torch.save(model.state_dict(), trained_model_file)
                best_val_criterion = val_SROCC  # update best val SROCC

     
    # Test
    if args.test_ratio > 0:
        # load best model for final test
        model.load_state_dict(torch.load(trained_model_file, map_location=device))
        model.eval()

        y_pred_test = np.zeros(len(test_index))
        y_test = np.zeros(len(test_index))
        L = 0

        with torch.no_grad():
            for i, (features, length, label) in enumerate(test_loader):
                y_test[i] = scale * label.item()

                features = features.to(device).float()
                features = torch.squeeze(features, dim=0)
                label = label.to(device).float()

                outputs = model(features)
                y_pred_test[i] = scale * (torch.mean(outputs).item())

                loss = criterion(outputs, label)
                L += loss.item()

        test_loss = L / (i + 1)
        test_SROCC = stats.spearmanr(y_pred_test, y_test)[0]
        test_KROCC = stats.kendalltau(y_pred_test, y_test)[0]

        beta_init = [np.max(y_test), np.min(y_test), np.mean(y_pred_test), 0.5]
        popt, _ = curve_fit(logistic_func, y_pred_test, y_test, p0=beta_init, maxfev=int(1e8))
        y_pred_test_logistic = logistic_func(y_pred_test, *popt)

        test_PLCC = stats.pearsonr(y_test, y_pred_test_logistic)[0]
        test_RMSE = np.sqrt(mean_squared_error(y_test, y_pred_test_logistic))

        logger.info(
            "Final Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(
                test_loss, test_SROCC, test_KROCC, test_PLCC, test_RMSE
            )
        )
    else:
        logger.info("Final Test skipped because args.test_ratio == 0")

    logger.reset()
