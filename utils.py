import torch
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import time
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score

def loss_fn(y_pred, y_true, classes, device):
    y_true = y_true.long()
    y_true = torch.eye(classes).to(device)[y_true]
    cross_entropy_loss = F.cross_entropy(y_pred, y_true)
    return cross_entropy_loss

def accuracy_fn(y_pred, y_true):
    predicted = torch.argmax(y_pred, 1)
    correct = (predicted == y_true).sum().item()
    total = y_true.size(0)
    return correct / total

def AA_fn(y_pred, y_true):
    class_number = y_pred.size(1)
    correct = np.zeros(class_number)
    total = np.zeros(class_number)
    y_pred = torch.argmax(y_pred, axis=-1)
    for i in range(class_number):
        total[i] = (y_true == i).sum().item()
        correct[i] = ((y_pred == y_true) * (y_true == i)).sum().item()

    return correct, total

def kappa_fn(y_pred, y_true):
    y_pred = np.argmax(y_pred, axis = -1)
    kappa = cohen_kappa_score(y_pred, y_true)
    return kappa

def train(train_data,train_label,batch_size,model,dataset_name,class_num,device,lr, epoches=200):

    last_number=int(len(train_data)%batch_size)
    train_data=torch.from_numpy(train_data.astype(np.float32)).to(device)
    train_label=torch.from_numpy(train_label.astype(np.float32)).to(device)
    train_dataset=TensorDataset(train_data,train_label)
    
    if last_number==1:
        train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
    else:
        train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches, eta_min=lr*0.01)

    Loss=99999
    for epoch in range(epoches):
        e_t=time.time()
        losss=0
        corrects=np.zeros(class_num)
        totals=np.zeros(class_num)
        epoch_preds=[]
        epoch_labels=[]
        model.train()
        for data,label in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, label, class_num, device)
            correct, total = AA_fn(outputs, label)
            loss.backward()
            optimizer.step()
            losss += loss.item()
            corrects += correct
            totals += total
            epoch_preds.append(outputs.detach().cpu())
            epoch_labels.append(label.detach().cpu())
        epoch_loss = losss
        epoch_accuracy = corrects.sum()/ (totals.sum() if totals.sum()>0 else 1)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        if Loss>epoch_loss:
            Loss=epoch_loss
            torch.save(model.state_dict(), './best_'+type(model).__name__+'_'+dataset_name+'_weights.pth')
            print(f'-------------save model at epoch {epoch+1 }---------------')
        print(f"Epoch {epoch+1}/{epoches}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, lr: {current_lr:.6f}, Epoch_time:{time.time()-e_t:.4f}")
        

def test(test_input_data, test_label, batch_size, model, dataset_name, num_class, device):

    model=model.to(device)
    state_dict = torch.load('./best_'+type(model).__name__+'_'+dataset_name+'_weights.pth')
    state_dict = {k: v for k, v in state_dict.items() if 'total_ops' not in k and 'total_params' not in k}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    test_input_data=torch.from_numpy(test_input_data.astype(np.float32))
    test_label=torch.from_numpy(test_label.astype(np.float32))
    test_dataset=TensorDataset(test_input_data,test_label)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    corrects=np.zeros(num_class)
    totals=np.zeros(num_class)
    output=[]
    true_label=[]
    with torch.no_grad():
        for data, labels in tqdm(test_loader):
            data=data.to(device)
            labels=labels.to(device)
            outputs = model(data)
            correct,total=AA_fn(outputs,labels)
            corrects+=correct
            totals+=total
            outputs=outputs.cpu().tolist()
            output+=outputs
            true_label+=labels.cpu().tolist()

    OA=corrects.sum()/totals.sum()
    output=np.array(output)
    kappa=kappa_fn(output,true_label)
    acc_class=np.divide(corrects,totals)
    AA=np.mean(acc_class)
    
    y_pred = np.argmax(output, axis=-1)
    y_true = np.array(true_label)
    cm = confusion_matrix(y_true, y_pred)
    
    print(OA)
    return np.argmax(output,axis=-1),OA,acc_class,AA,kappa,cm
