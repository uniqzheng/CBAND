import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo
import skvideo.io
from PIL import Image
import os
import cv2
import h5py
import numpy as np
np.float = np.float64
np.int = np.int_
import random
from argparse import ArgumentParser
from matplotlib import pyplot as plot
from scipy.special import gamma
import warnings
import time
warnings.filterwarnings("ignore")

def estimate_ggd_param(vec):
    """Estimate GGD parameter.

    :param vec: The vector that we want to approximate its parameter.
    :type vec: np.ndarray
    """
    gam = np.arange(0.2, 10 + 0.001, 0.001)
    r_gam = (gamma(1.0 / gam) * gamma(3.0 / gam) / (gamma(2.0 / gam) ** 2))

    sigma_sq = np.mean(vec ** 2)
    sigma = np.sqrt(sigma_sq)
    E = np.mean(np.abs(vec))
    rho = sigma_sq / E ** 2

    differences = np.abs(rho - r_gam)
    array_position = np.argmin(differences)
    gamparam = gam[array_position]

    return gamparam, sigma

def gen_MSCNfeature(imdist):
    mu = cv2.GaussianBlur(
        imdist, (7, 7), 7 / 6, borderType=cv2.BORDER_CONSTANT)
    mu_sq = mu * mu
    sigma = cv2.GaussianBlur(
        imdist * imdist, (7, 7), 7 / 6, borderType=cv2.BORDER_CONSTANT)
    sigma = np.sqrt(np.abs((sigma - mu_sq)))
    structdis = (imdist - mu) / (sigma + 1)

    alpha, overallstd = estimate_ggd_param(structdis)
    return alpha, overallstd ** 2

def gen_MSCNfeature_loop(batch_maps):
    all_maps = np.reshape(batch_maps,(-1,batch_maps.shape[-2],batch_maps.shape[-1]))
    all_feat = np.zeros([all_maps.shape[0],2])
    for i in range(all_maps.shape[0]):
        cur_map = all_maps[i]
        feat1, feat2 = gen_MSCNfeature(cur_map)
        all_feat[i] = [feat1, feat2]
    batch_feat = np.reshape(all_feat,(batch_maps.shape[0],batch_maps.shape[1],2))
    return batch_feat


class VideoDataset(Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, videos_dir, video_names, score, video_format='RGB', width=None, height=None):

        super(VideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.video_names = video_names
        self.score = score
        self.format = video_format
        self.width = width
        self.height = height
        self.gaussian_filter = np.array([[1/16.,1/8.,1/16.],[1/8.,1/4.,1/8.],[1/16.,1/8.,1/16.]])

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
       
        # for Konvid
        prefix = video_name.split('_')[0]
        true_video_name = prefix + '.mp4'
        
        video_data = skvideo.io.vread(os.path.join(self.videos_dir, true_video_name))
        video_score = self.score[idx]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        video_length = video_data.shape[0]
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]
        transformed_video = torch.zeros([video_length, video_channel,  video_height, video_width])
        for frame_idx in range(video_length):
            frame = video_data[frame_idx]
            frame = Image.fromarray(frame)
            frame = transform(frame)
            transformed_video[frame_idx] = frame

        sample = {'video': transformed_video,
                  'score': video_score}

        return sample

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.vgg16 = models.vgg16(pretrained=True)

        self.features = nn.Sequential(*list(self.vgg16.features.children()))     # stage5
        self.stage4 = nn.Sequential(*list(self.vgg16.features.children()))[:-7]
        self.stage3 = nn.Sequential(*list(self.vgg16.features.children()))[:-14]
        self.stage2 = nn.Sequential(*list(self.vgg16.features.children()))[:-21]
        self.stage1 = nn.Sequential(*list(self.vgg16.features.children()))[:-26]
        
        self.mscnfeat = gen_MSCNfeature_loop

    def forward(self, x):
        x1 = self.stage1(x)
        x1_np = x1.cpu().numpy()
        x1_mscn_out = self.mscnfeat(x1_np)
        torch.cuda.empty_cache()

        x2 = self.stage2(x)
        x2_np = x2.cpu().numpy()
        x2_mscn_out = self.mscnfeat(x2_np)
        torch.cuda.empty_cache()

        x3 = self.stage3(x)
        x3_np = x3.cpu().numpy()
        x3_mscn_out = self.mscnfeat(x3_np)
        torch.cuda.empty_cache()

        x4 = self.stage4(x)
        x4_np = x4.cpu().numpy()
        x4_mscn_out = self.mscnfeat(x4_np)
        torch.cuda.empty_cache()

        x5 = self.features(x)
        x5_np = x5.cpu().numpy()
        x5_mscn_out = self.mscnfeat(x5_np)
        torch.cuda.empty_cache()

        x1_mscn_out = torch.from_numpy(x1_mscn_out).to(device).float()
        x2_mscn_out = torch.from_numpy(x2_mscn_out).to(device).float()
        x3_mscn_out = torch.from_numpy(x3_mscn_out).to(device).float()
        x4_mscn_out = torch.from_numpy(x4_mscn_out).to(device).float()
        x5_mscn_out = torch.from_numpy(x5_mscn_out).to(device).float()

        return x1_mscn_out,x2_mscn_out,x3_mscn_out,x4_mscn_out,x5_mscn_out


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()

        self.resnet50 = models.resnet50(pretrained=True)


        self.stage4 = nn.Sequential(*list(self.resnet50.children())[:-2])
        self.stage3 = nn.Sequential(*list(self.resnet50.children())[:-3])
        self.stage2 = nn.Sequential(*list(self.resnet50.children())[:-4])
        self.stage1 = nn.Sequential(*list(self.resnet50.children())[:-5])

        for p in self.stage4.parameters():
            p.requires_grad = False
        for p in self.stage3.parameters():
            p.requires_grad = False
        for p in self.stage2.parameters():
            p.requires_grad = False
        for p in self.stage1.parameters():
            p.requires_grad = False

        self.mscnfeat = gen_MSCNfeature_loop

    def forward(self, x):

        x1 = self.stage1(x)
        x1_np = x1.cpu().numpy()
        x1_mscn_out = self.mscnfeat(x1_np)

        x2 = self.stage2(x)
        x2_np = x2.cpu().numpy()
        x2_mscn_out = self.mscnfeat(x2_np)

        x3 = self.stage3(x)
        x3_np = x3.cpu().numpy()
        x3_mscn_out = self.mscnfeat(x3_np)

        x4 = self.stage4(x)
        x4_np = x4.cpu().numpy()
        x4_mscn_out = self.mscnfeat(x4_np)

        x1_mscn_out = torch.from_numpy(x1_mscn_out).to(device).float()
        x2_mscn_out = torch.from_numpy(x2_mscn_out).to(device).float()
        x3_mscn_out = torch.from_numpy(x3_mscn_out).to(device).float()
        x4_mscn_out = torch.from_numpy(x4_mscn_out).to(device).float()


        return x1_mscn_out,x2_mscn_out,x3_mscn_out,x4_mscn_out


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

def get_features(video_data, frame_batch_size=64, device='cuda'):
    """feature extraction"""
    extractor = Resnet50().to(device)
    # extractor = VGG16().to(device)
    # extractor = CustomDataParallel(extractor)
    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    output3 = torch.Tensor().to(device)
    output4 = torch.Tensor().to(device)
    # output5 = torch.Tensor().to(device)
    extractor.eval()
    with torch.no_grad():
        
        while frame_end < video_length:
            batch = video_data[frame_start:frame_end].to(device)
            features1, features2, features3, features4 = extractor(batch)
            output1 = torch.cat((output1, features1), 0)
            output2 = torch.cat((output2, features2), 0)
            output3 = torch.cat((output3, features3), 0)
            output4 = torch.cat((output4, features4), 0)
            # output5 = torch.cat((output5, features5), 0)
            frame_end += frame_batch_size
            frame_start += frame_batch_size

        last_batch = video_data[frame_start:video_length].to(device)
        features1, features2, features3, features4 = extractor(last_batch)
        output1 = torch.cat((output1, features1), 0)
        output2 = torch.cat((output2, features2), 0)
        output3 = torch.cat((output3, features3), 0)
        output4 = torch.cat((output4, features4), 0)
        # output5 = torch.cat((output5, features5), 0)

        output1 = output1.squeeze()
        output2 = output2.squeeze()
        output3 = output3.squeeze()
        output4 = output4.squeeze()
        # output5 = output5.squeeze()

    return output1, output2, output3, output4


if __name__ == "__main__":

    parser = ArgumentParser(description='"Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50 or VGG16')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--database', default='KoNViD-1k', type=str,
                        help='database name (default: KoNViD-1k)')
    parser.add_argument('--frame_batch_size', type=int, default=64,
                        help='frame batch size for feature extraction (default: 64)')

    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args()

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    if args.database == 'KoNViD-1k':
        videos_dir = '/workspace/dataset/KoNViD_1k_videos/'  # videos dir
        datainfo = '/workspace/CBAND/data/KoNViD-1kinfo.mat'  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
        features_dir1 = '/workspace/CBAND/features/CNN_features_KoNViD-1k_Resnet50_stage1_ggd2/'
        features_dir2 = '/workspace/CBAND/features/CNN_features_KoNViD-1k_Resnet50_stage2_ggd2/'
        features_dir3 = '/workspace/CBAND/features/CNN_features_KoNViD-1k_Resnet50_stage3_ggd2/'
        features_dir4 = '/workspace/CBAND/features/CNN_features_KoNViD-1k_Resnet50_stage4_ggd2/'
    if args.database == 'LIVE_Banding':
        videos_dir = '/workspace/dataset/LIVE_BANDING/final_set/'
        datainfo = '/workspace/CBAND/data/LIVE_Banding_contentaware.mat'
        features_dir1 = '/workspace/CBAND/features/CNN_features_LIVE_Banding_Resnet50_stage1_ggd2/'
        features_dir2 = '/workspace/CBAND/features/CNN_features_LIVE_Banding_Resnet50_stage2_ggd2/'
        features_dir3 = '/workspace/CBAND/features/CNN_features_LIVE_Banding_Resnet50_stage3_ggd2/'
        features_dir4 = '/workspace/CBAND/features/CNN_features_LIVE_Banding_Resnet50_stage4_ggd2/'
        # features_dir5 = '/workspace/CBAND/features/CNN_features_LIVE_Banding_Vgg16_stage5_ggd2/'

    if not os.path.exists(features_dir1):
        os.makedirs(features_dir1)
    if not os.path.exists(features_dir2):
        os.makedirs(features_dir2)
    if not os.path.exists(features_dir3):
        os.makedirs(features_dir3)
    if not os.path.exists(features_dir4):
        os.makedirs(features_dir4)
    # if not os.path.exists(features_dir5):
    #     os.makedirs(features_dir5)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")

    Info = h5py.File(datainfo, 'r')
    video_names = [Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode() for i in
                   range(len(Info['video_names'][0, :]))]
    scores = Info['scores'][0, :]
    video_format = Info['video_format'][()].tobytes()[::2].decode()
    width = int(Info['width'][0])
    height = int(Info['height'][0])
    
    
    dataset = VideoDataset(videos_dir, video_names, scores, video_format, width, height)
    
    for i in range(len(dataset)):
        
        current_data = dataset[i]
        current_video = current_data['video']
        current_score = current_data['score']
        print('Video {}: length {}'.format(i, current_video.shape[0]))
        start_time = time.time()
        features1, features2, features3, features4 = get_features(current_video,args.frame_batch_size,device)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("feature extraction consume time: ", elapsed_time, "seconds")

        features1 = features1.permute(0, 2, 1).contiguous().view(features1.shape[0], -1)
        features2 = features2.permute(0, 2, 1).contiguous().view(features2.shape[0], -1)
        features3 = features3.permute(0, 2, 1).contiguous().view(features3.shape[0], -1)
        features4 = features4.permute(0, 2, 1).contiguous().view(features4.shape[0], -1)
        # # features5 = features5.permute(0, 2, 1).contiguous().view(features5.shape[0], -1)

        print('Time of feature calculation: ', end_time - start_time)
        if len(features1.shape) == 2 and features1.shape[1] == 512:
            np.save(features_dir1 + str(i) + '_resnet50_stage1_ggd2', features1.to('cpu').numpy())
            np.save(features_dir1 + str(i) + '_score', current_score)
        else:
            print("size of features is wrong")
            break
        if len(features2.shape) == 2 and features2.shape[1] == 1024:
            np.save(features_dir2 + str(i) + '_resnet50_stage2_ggd2', features2.to('cpu').numpy())
            np.save(features_dir2 + str(i) + '_score', current_score)
        else:
            print("size of features is wrong")
            break
        if len(features3.shape) == 2 and features3.shape[1] == 2048:
            np.save(features_dir3 + str(i) + '_resnet50_stage3_ggd2', features3.to('cpu').numpy())
            np.save(features_dir3 + str(i) + '_score', current_score)
        else:
            print("size of features is wrong")
            break
        if len(features4.shape) == 2 and features4.shape[1] == 4096:
            np.save(features_dir4 + str(i) + '_resnet50_stage4_ggd2', features4.to('cpu').numpy())
            np.save(features_dir4 + str(i) + '_score', current_score)
        else:
            print("size of features is wrong")
            break
        
        # if len(features1.shape) == 2 and features1.shape[1] == 128:
        #     np.save(features_dir1 + str(i) + '_vgg16_stage1_ggd2', features1.to('cpu').numpy())
        #     np.save(features_dir1 + str(i) + '_score', current_score)
        # else:
        #     print("size of features is wrong")
        #     break
        # if len(features2.shape) == 2 and features2.shape[1] == 256:
        #     np.save(features_dir2 + str(i) + '_vgg16_stage2_ggd2', features2.to('cpu').numpy())
        #     np.save(features_dir2 + str(i) + '_score', current_score)
        # else:
        #     print("size of features is wrong")
        #     break
        # if len(features3.shape) == 2 and features3.shape[1] == 512:
        #     np.save(features_dir3 + str(i) + '_vgg16_stage3_ggd2', features3.to('cpu').numpy())
        #     np.save(features_dir3 + str(i) + '_score', current_score)
        # else:
        #     print("size of features is wrong")
        #     break
        # if len(features4.shape) == 2 and features4.shape[1] == 1024:
        #     np.save(features_dir4 + str(i) + '_vgg16_stage4_ggd2', features4.to('cpu').numpy())
        #     np.save(features_dir4 + str(i) + '_score', current_score)
        # else:
        #     print("size of features is wrong")
        #     break
        # if len(features5.shape) == 2 and features5.shape[1] == 1024:
        #     np.save(features_dir5 + str(i) + '_vgg16_stage5_ggd2', features5.to('cpu').numpy())
        #     np.save(features_dir5 + str(i) + '_score', current_score)
        # else:
        #     print("size of features is wrong")
        #     break


