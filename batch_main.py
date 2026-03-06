import torch
from preprocess import loadData, get_location, set_seed, PatchDataset
from utils import loss_fn, AA_fn, kappa_fn, confusion_matrix
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
from MS2GCAN import MS2GCAN
import time
from torch.utils.data import DataLoader

device = torch.device("cuda:0")
torch.cuda.set_device(device)

seed_list = [0]
dataset_list = ['PU', 'HU', 'WHLK']
dataset_name = dataset_list[2] # 0'PU', 1'HU', 2'WHLK'
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
# test_only = True
test_only = False

data, labels_TE, labels_TR, class_num = loadData(dataset_name)
H, W, S = data.shape
input_dim = S
data = np.reshape(data, [H * W, -1])
minMax = preprocessing.StandardScaler()
data = minMax.fit_transform(data)
data = np.reshape(data, [H, W, -1])  # 使用-1自动推断维度
train_location, train_label = get_location(labels_TR)
test_location, test_label = get_location(labels_TE)

train_dataset_full = PatchDataset(data, train_location, train_label, data_size)

def collate_fn(batch):
    patches, labels = zip(*batch)  # 每个patch: (l*l, S)
    patches = torch.stack(patches, dim=0)  # (B, l*l, S)
    labels = torch.tensor(labels, dtype=torch.float32)
    return patches, labels

all_results = []
for seed_idx, seed in enumerate(seed_list):
    print(f"\n{'='*60}")
    print(f"开始第 {seed_idx + 1}/{len(seed_list)} 次实验，种子: {seed}")
    print(f"{'='*60}")
    
    # 设置当前种子
    # set_seed(seed)

    train_loader_dynamic = DataLoader(train_dataset_full, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = MS2GCAN(T=T, img_size=data_size, num_cls=class_num, input_dim=input_dim, hidden=hidden, gcn_layers=gcn_layers, K_hop=K_hop, use_cupy=True).to(device)

    if not test_only:
        optimiser = torch.optim.Adam(model.parameters(),lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs, eta_min=lr*0.01)
        best_loss = 1e9
        for epoch in range(epochs):
            model.train()
            e_t=time.time()
            total_loss = 0.0
            corrects = np.zeros(class_num)
            totals = np.zeros(class_num)
            for batch_x, batch_y in train_loader_dynamic:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimiser.zero_grad()
                outputs = model(batch_x)
                loss = loss_fn(outputs, batch_y, class_num, device)
                loss.backward()
                optimiser.step()
                total_loss += loss.item()
                correct, total = AA_fn(outputs, batch_y)
                corrects += correct
                totals += total
            oa = corrects.sum() / (totals.sum() if totals.sum() > 0 else 1)
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}/{epochs} Loss {total_loss:.4f} OA {oa:.4f} LR {current_lr:.6f}, ET {time.time()-e_t:.4f}")

            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(model.state_dict(), f'./best_{type(model).__name__}_{dataset_name}_weights.pth')
                print('保存最佳模型')

    print(f"开始测试模型...")
    test_dataset_dynamic = PatchDataset(data, test_location, test_label, data_size)
    test_loader_dynamic = DataLoader(test_dataset_dynamic, batch_size=batch_size*16, shuffle=False, collate_fn=collate_fn)
    state_dict = torch.load(f'./best_{type(model).__name__}_{dataset_name}_weights.pth')
    state_dict = {k: v for k, v in state_dict.items() if 'total_ops' not in k and 'total_params' not in k}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    corrects = np.zeros(class_num)
    totals = np.zeros(class_num)
    outputs_all = []
    labels_all = []
    for batch_x, batch_y in tqdm(test_loader_dynamic, desc="Testing"):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        out = model(batch_x)
        correct, total = AA_fn(out, batch_y)
        corrects += correct
        totals += total
        outputs_all.append(out.cpu().detach().numpy())
        labels_all.append(batch_y.cpu().numpy())
    outputs_np = np.concatenate(outputs_all, axis=0)
    labels_np = np.concatenate(labels_all, axis=0)
    OA = corrects.sum() / totals.sum()
    acc_class = corrects / totals
    AA = acc_class.mean()
    kappa = kappa_fn(outputs_np, labels_np)
    cm = confusion_matrix(labels_np, np.argmax(outputs_np, axis=1))
    
    print(f"测试集结果: OA {OA:.4f} AA {AA:.4f} Kappa {kappa:.4f}")
        
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
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

# 所有实验完成后，计算并保存统计结果
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