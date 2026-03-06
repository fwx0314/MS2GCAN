import torch
from preprocess import loadData, get_location, patch_data, set_seed
import numpy as np
from MS2GCAN import MS2GCAN
from utils import train, test
from sklearn import preprocessing

device = torch.device("cuda:0")
torch.cuda.set_device(device)

seed_list = [0]
dataset_list = ['PU', 'HU', 'WHLK']
dataset_name = dataset_list[0] # 0'PU', 1'HU', 2'WHLK'
if dataset_name == 'PU':
    data_size = 19
    T = 3
    gcn_layers = 2
    K_hop = 2
    hidden = 32
    batch_size = 64
if dataset_name == 'HU':
    data_size = 13
    T = 3
    gcn_layers = 2
    K_hop = 1
    hidden = 96
    batch_size = 32
if dataset_name == 'WHLK':
    data_size = 15
    T = 3
    gcn_layers = 2
    K_hop = 1
    hidden = 48
    batch_size = 32

epochs = 200
lr = 0.0001
test_only = True
test_only = False

data, labels_TE, labels_TR, class_num = loadData(dataset_name)
H, W, S = data.shape
input_dim = S
data = np.reshape(data, [H * W, -1])
minMax = preprocessing.StandardScaler()
data = minMax.fit_transform(data)
data = np.reshape(data, [H, W, -1])
train_location, train_label = get_location(labels_TR)
test_location, test_label = get_location(labels_TE)
train_data = patch_data(data, data_size, train_location)
test_data = patch_data(data, data_size, test_location)

all_results = []
for seed_idx, seed in enumerate(seed_list):
    print(f"\n{'='*60}")
    print(f"开始第 {seed_idx + 1}/{len(seed_list)} 次实验，种子: {seed}")
    print(f"{'='*60}")
    
    # 设置当前种子
    # set_seed(seed)

    model = MS2GCAN(T=T, img_size=data_size, num_cls=class_num, input_dim=input_dim, hidden=hidden, gcn_layers=gcn_layers, K_hop=K_hop, use_cupy=True).to(device)

    if not test_only:
        train(train_data, train_label, batch_size, model, dataset_name, class_num, device, lr, epochs)

    _, OA, acc_class, AA, kappa, cm = test(test_data, test_label, batch_size, model, dataset_name, class_num, device)

    # 计算每类的精确率、召回率、F1分数
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)
        
    # 收集实验结果
    result = {
        'seed': seed,
        'OA': OA,
        'AA': AA,
        'kappa': kappa,
        'acc_class': acc_class,
        'cm': cm,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
    all_results.append(result)
    
    print(f"种子 {seed} 的实验完成:")
    print(f"  OA: {OA:.4f}")
    print(f"  AA: {AA:.4f}")
    print(f"  Kappa: {kappa:.4f}")

print(f"\n{'='*60}")
print("所有实验完成！正在计算统计结果...")
print(f"{'='*60}")

# 计算统计结果
OA_list = [result['OA'] for result in all_results]
AA_list = [result['AA'] for result in all_results]
kappa_list = [result['kappa'] for result in all_results]

OA_mean = np.mean(OA_list)
OA_std = np.std(OA_list)
AA_mean = np.mean(AA_list)
AA_std = np.std(AA_list)
kappa_mean = np.mean(kappa_list)
kappa_std = np.std(kappa_list)

# 计算每类精度的统计
acc_class_list = np.array([result['acc_class'] for result in all_results])
acc_class_mean = np.mean(acc_class_list, axis=0)
acc_class_std = np.std(acc_class_list, axis=0)

print(f"\n最终统计结果:")
print(f"OA: {OA_mean:.6f} ± {OA_std:.6f}")
print(f"AA: {AA_mean:.6f} ± {AA_std:.6f}")
print(f"Kappa: {kappa_mean:.6f} ± {kappa_std:.6f}")
