from scipy.io import loadmat
import h5py
import numpy as np
from tqdm import tqdm
import torch
import random
import math
import os
from torch.utils.data import Dataset

def loadData(Data):
    # 读入数据
    if Data == 'IP':
        data = loadmat('./data/Indian_pines_corrected.mat')['indian_pines_corrected']
        labels_TE = loadmat('./data/IndianPine.mat')['TE']
        labels_TR = loadmat('./data/IndianPine.mat')['TR']
        class_num = 16
    elif Data == 'PU':
        data = loadmat('./data/patch_data/PU/PaviaU.mat')['paviaU']
        labels_TE = loadmat('./data/patch_data/PU/DS_PaviaU_gt_TE.mat')['y']
        labels_TR = loadmat('./data/patch_data/PU/DS_PaviaU_gt_TR.mat')['y']
        class_num = 9
    elif Data == 'HU':
        data = loadmat('./data/patch_data/HU/Houston2013.mat')['HSI']
        labels_TE = loadmat('./data/patch_data/HU/TSLabel.mat')['TSLabel']
        labels_TR = loadmat('./data/patch_data/HU/TRLabel.mat')['TRLabel']
        class_num = 15
    elif Data == 'YC':
        data = loadmat('./data/patch_data/Yancheng/data_hsi.mat')['data']
        labels_TE = loadmat('./data/patch_data/Yancheng/test_label.mat')['test_label']
        labels_TR = loadmat('./data/patch_data/Yancheng/train_label.mat')['train_label']
        class_num = 18
    elif Data == 'LN':
        data = loadmat('./data/patch_data/LiaoNing/LN01_HSI.mat')['HSI']
        labels_TE = loadmat('./data/patch_data/LiaoNing/LN_TE.mat')['gt']
        labels_TR = loadmat('./data/patch_data/LiaoNing/LN_TR.mat')['gt']
        class_num = 10
    elif Data == 'WHLK':
        data = loadmat('./data/patch_data/WHULK/WHU_Hi_LongKou.mat')['WHU_Hi_LongKou']
        labels_TE = loadmat('./data/patch_data/WHULK/LK_Test.mat')['WHU_Hi_LongKou_gt']
        labels_TR = loadmat('./data/patch_data/WHULK/LK_Train.mat')['WHU_Hi_LongKou_gt']
        class_num = 9

    else:
        if Data == "KSC":
            data = loadmat('./data/patch_data/KSC/KSC.mat')['KSC']
            label = loadmat('./data/patch_data/KSC/KSC_gt.mat')['KSC_gt']
            class_num = 13
            class_numbers = [0 for i in range(int(class_num))]
            for items in list(label):
                for item in items:
                    if item!=0:
                        class_numbers[item-1] += 1           
            class_numbers_train = [math.ceil(0.001 * class_numbers[i]) for i in range(class_num)]
        elif Data == "WHHC":
            data_path = os.path.join(r'/home/ubuntu/dataset_RS/classification')
            data = loadmat(os.path.join(data_path, 'WHU-Hi-HanChuan', 'WHU_Hi_HanChuan.mat'))['WHU_Hi_HanChuan']
            label = loadmat(os.path.join(data_path, 'WHU-Hi-HanChuan', 'WHU_Hi_HanChuan_gt.mat'))['WHU_Hi_HanChuan_gt']
            class_num = 16
            class_numbers = [0 for i in range(int(class_num))]
            for items in list(label):
                for item in items:
                    if item != 0:
                        class_numbers[item - 1] += 1
            # class_numbers_train = [math.ceil(0.001 * class_numbers[i]) for i in range(class_num)]
            class_numbers_train = [10 for i in range(class_num)]
        elif Data == "WHLK":
            data_path = os.path.join(r'/home/ubuntu/dataset_RS/classification')
            data = loadmat(os.path.join(data_path, 'WHU-Hi-LongKou', 'WHU_Hi_LongKou.mat'))['WHU_Hi_LongKou']
            label = loadmat(os.path.join(data_path, 'WHU-Hi-LongKou', 'WHU_Hi_LongKou_gt.mat'))['WHU_Hi_LongKou_gt']
            class_num = 9
            class_numbers = [0 for i in range(int(class_num))]
            for items in list(label):
                for item in items:
                    if item != 0:
                        class_numbers[item - 1] += 1
            # class_numbers_train = [math.ceil(0.001 * class_numbers[i]) for i in range(class_num)]
            class_numbers_train = [10 for i in range(class_num)]
        elif Data == "PC":
            data = loadmat('./data/patch_data/PC/pavia.mat')['HSI_original']
            label = loadmat('./data/patch_data/PC/pavia_gt.mat')['Data_gt']
            class_num = 9
            #class_numbers_train = [10 for i in range(class_num)]
            class_numbers = [0 for i in range(int(class_num))]
            for items in list(label):
                for item in items:
                    if item!=0:
                        class_numbers[item-1] += 1           
            class_numbers_train = [math.ceil(0.001 * class_numbers[i]) for i in range(class_num)]
        elif Data == 'PU_normal':
            data = loadmat('./data/patch_data/PU/PaviaU.mat')['paviaU']
            label = loadmat('./data/patch_data/PU/PaviaU_gt.mat')['paviaU_gt']
            class_num = 9
            class_numbers = [0 for i in range(int(class_num))]
            for items in list(label):
                for item in items:
                    if item!=0:
                        class_numbers[item-1] += 1           
            class_numbers_train = [math.ceil(0.002 * class_numbers[i]) for i in range(class_num)]
        elif Data == 'IP_normal':
            data = loadmat('./data/patch_data/IP/Indian_pines_corrected.mat')['indian_pines_corrected']
            label = loadmat('./data/patch_data/IP/Indian_pines_gt.mat')['indian_pines_gt']
            class_num=16
            class_numbers = [0 for i in range(int(class_num))]
            for items in list(label):
                for item in items:
                    if item!=0:
                        class_numbers[item-1] += 1           
            class_numbers_train = [math.ceil(0.02 * class_numbers[i]) for i in range(class_num)]
        elif Data == 'SA':
            data = loadmat('./data/patch_data/SA/Salinas_corrected.mat')['salinas_corrected']
            label = loadmat('./data/patch_data/SA/Salinas_gt.mat')['salinas_gt']
            class_num = 16
            class_numbers = [0 for i in range(int(class_num))]
            for items in list(label):
                for item in items:
                    if item!=0:
                        class_numbers[item-1] += 1           
            class_numbers_train = [math.ceil(0.001 * class_numbers[i]) for i in range(class_num)]
        elif Data == 'HU2018':
            data = h5py.File('./data/patch_data/HU/HoustonU2018.mat')['houstonU'][:]
            label = h5py.File('./data/patch_data/HU/HoustonU_gt2018.mat')['houstonU_gt'][:]
            data = np.transpose(data, axes=[2, 1, 0])
            label = np.transpose(label, axes=[1, 0])
            class_num = 20
            #class_numbers_train = [50 for i in range(int(class_num))]
            class_numbers = [0 for i in range(int(class_num))]
            for items in list(label):
                for item in items:
                    if item!=0:
                        class_numbers[item-1] += 1
            class_numbers_train = [math.ceil(0.001 * class_numbers[i]) for i in range(class_num)]
            for i in range(len(class_numbers_train)):
                if class_numbers_train[i]<10:
                    class_numbers_train[i]=10
                if class_numbers_train[i]>100:
                    class_numbers_train[i]=100
        elif Data == 'HU2013':
            data = loadmat('./data/patch_data/HU/Houston.mat')['Houston']
            label = loadmat('./data/patch_data/HU/Houston_gt.mat')['Houston_gt']
            class_num = 15
            class_numbers = [0 for i in range(int(class_num))]
            for items in list(label):
                for item in items:
                    if item!=0:
                        class_numbers[item-1] += 1           
            class_numbers_train = [math.ceil(0.05 * class_numbers[i]) for i in range(class_num)]
        elif Data == 'BS':
            data = loadmat('./data/patch_data/BS/Botswana.mat')['Botswana']
            label = loadmat('./data/patch_data/BS/Botswana_gt.mat')['Botswana_gt']
            class_num = 14
            class_numbers = [0 for i in range(int(class_num))]
            for items in list(label):
                for item in items:
                    if item!=0:
                        class_numbers[item-1] += 1           
            class_numbers_train = [math.ceil(0.01 * class_numbers[i]) for i in range(class_num)]
        elif Data == 'LN01':
            data = loadmat('./data/patch_data/LiaoNing/LN01_HSI.mat')['HSI']
            label = loadmat('./data/patch_data/LiaoNing/LN01_gt.mat')['gt']
            class_num = 10
            class_numbers = [0 for i in range(int(class_num))]
            for items in list(label):
                for item in items:
                    if item!=0:
                        class_numbers[item-1] += 1           
            class_numbers_train = [math.ceil(0.01 * class_numbers[i]) for i in range(class_num)]
        
        print(class_numbers_train)
        if os.path.exists(Data+'_TE.npy'):
            labels_TE=np.load(Data+'_TE.npy')
            labels_TR=np.load(Data+'_TR.npy')
            print('Dataset Loaded')
        else:
            labels_TE=np.copy(label)
            for i in range(len(class_numbers_train)):
                target_value = i+1
                replace_count=class_numbers_train[i]
                indices = np.where(labels_TE == target_value)
                random_indices = np.random.choice(range(len(indices[0])), replace_count, replace=False)
                for j in random_indices:
                    row, col = indices[0][j], indices[1][j]
                    labels_TE[row, col] = 0
            labels_TR=label-labels_TE
            np.save(Data+'_TE.npy',labels_TE)
            np.save(Data+'_TR.npy',labels_TR)
            print('Dataset Spilted')

    return data, labels_TE, labels_TR, class_num

def get_location(label):
    location=[]
    labels=[]
    H,W=label.shape
    for i in range(H):
        for j in range(W):
            if label[i,j]!=0:
                location.append([i,j])
                labels.append(label[i,j]-1)
    location=np.array(location)
    labels=np.array(labels)
    return location,labels

def patch_data(data, l, location):
    # 获取数据的维度
    H, W, S = data.shape
    patch_data = []
    
    # 当patch边长l为1时，直接提取每个位置的数据
    if l == 1:
        for idx in tqdm(list(location)):
            i, j = idx
            # 将每个位置的数据添加到patch_data中，并重塑为一维数组
            patch_data.append(data[i, j, :].reshape(-1))
    else:
        # 当patch边长大于1时，提取指定大小的patch
        for idx in tqdm(list(location)):
            mask = np.float32(np.zeros([l, l, S]))
            i, j = idx
            # 计算patch的边界位置
            up = i - int(l / 2)
            down = i + int(l / 2)
            left = j - int(l / 2)
            right = j + int(l / 2)
            # 确保patch的边界位置不会超出数据的范围
            up = 0 if up < 0 else up
            left = 0 if left < 0 else left
            down = H - 1 if down > H - 1 else down
            right = W - 1 if right > W - 1 else right
            # 将数据复制到mask中，以形成完整的patch
            mask[int(l / 2) - (i - up):int(l / 2) + down - i + 1, int(l / 2) - (j - left):int(l / 2) + right - j + 1, :] = data[up:down + 1, left:right + 1, :]
            patch_data.append(mask)
    print('Data patched.')
    # 将patch_data转换为numpy数组，并重塑为指定的维度
    patch_data = np.float32(np.array(patch_data))
    patch_data = patch_data.reshape(-1, l*l, S)
    return patch_data


class PatchDataset(Dataset):
    """按需动态提取patch，避免一次性预计算占用大量内存。
    返回形状：(patch_size*patch_size, S) 的patch以及对应标签。
    """
    def __init__(self, data: np.ndarray, locations: np.ndarray, labels: np.ndarray, patch_size: int):
        self.data = data  # H W S
        self.locations = locations  # N 2
        self.labels = labels  # N
        self.patch_size = patch_size
        self.H, self.W, self.S = data.shape
        self.half = patch_size // 2
        assert patch_size % 2 == 1, "当前实现要求patch_size为奇数"  # 与原始逻辑一致
    def __len__(self):
        return len(self.locations)
    def _extract_patch(self, i: int, j: int):
        l = self.patch_size
        mask = np.zeros((l, l, self.S), dtype=np.float32)
        up = max(0, i - self.half)
        down = min(self.H - 1, i + self.half)
        left = max(0, j - self.half)
        right = min(self.W - 1, j + self.half)
        mask[self.half - (i - up): self.half + (down - i) + 1, self.half - (j - left): self.half + (right - j) + 1, :] = \
            self.data[up:down + 1, left:right + 1, :]
        patch = mask.reshape(-1, self.S)  # (l*l, S)
        return patch
    def __getitem__(self, idx):
        i, j = self.locations[idx]
        patch = self._extract_patch(int(i), int(j))
        label = self.labels[idx]
        return torch.from_numpy(patch).float(), torch.tensor(label, dtype=torch.float32)

def set_seed(seed):
    random.seed(seed) # python的随机性
    np.random.seed(seed) # np的随机性
    torch.manual_seed(seed) # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed) # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed) # torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True  # 启用确定性算法
    torch.backends.cudnn.benchmark = False  # 禁用benchmark优化以确保确定性
    # torch.backends.cudnn.enabled = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(True)  # 可选：启用完全确定性算法
    os.environ['PYTHONHASHSEED'] = str(seed)