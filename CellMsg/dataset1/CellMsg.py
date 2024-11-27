from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import torch
from torch_geometric.nn.conv import GCNConv
from sklearn.metrics import matthews_corrcoef
from scipy.sparse import csr_matrix
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, random_split, TensorDataset
from scipy import interp
import os
# import KTBoost.KTBoost as KB
# import gpboost as gb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

proteinFeature_L = pd.read_csv("/home/jby2/XH/CellMsg/dataset1/ligand_res_fea.csv", header=None, index_col=None)
proteinFeature_R = pd.read_csv("/home/jby2/XH/CellMsg/dataset1/receptor_res_fea.csv", header=None, index_col=None)
interaction = pd.read_csv("/home/jby2/XH/CellMsg/dataset1/ligand-receptor-interaction.csv", header=None, index_col=None, skiprows=1)

interaction = interaction.drop(columns=interaction.columns[0]).to_numpy()
proteinFeature_L = proteinFeature_L.drop(columns=proteinFeature_L.columns[0]).to_numpy()
proteinFeature_R = proteinFeature_R.drop(columns=proteinFeature_R.columns[0]).to_numpy()


L_R_fea = np.vstack((proteinFeature_L, proteinFeature_R))
# proteinFeature_R = torch.tensor(proteinFeature_R, dtype=torch.float32)
# ProteinFeature_L = torch.tensor(proteinFeature_L, dtype=torch.float32)
L_R_fea = torch.tensor(L_R_fea, dtype=torch.float32)


whole_interaction = np.zeros((interaction.shape[0] + interaction.shape[1], interaction.shape[0] + interaction.shape[1]))

whole_interaction[:interaction.shape[0], interaction.shape[0]:] = interaction
# whole_interaction[interaction.shape[0]:, :interaction.shape[0]] = interaction.T  

interaction1 = csr_matrix(whole_interaction)
row, col = interaction1.nonzero()
edge_index = np.vstack([row, col])
edge_index = torch.from_numpy(edge_index)



def Splicing_data(interaction):
    positive_feature = []
    negative_feature = []
    for i in range(np.shape(interaction)[0]):
        for j in range(np.shape(interaction)[1]):
            # temp = np.append(proteinFeature_L[i], proteinFeature_R[j])
            if int(interaction[i][j]) == 1:
                positive_feature.append((i, j))
            elif int(interaction[i][j]) == 0:
                negative_feature.append((i, j))

    negative_sample_index = np.random.choice(np.arange(len(negative_feature)), size=len(positive_feature),
                                             replace=False)
    negative_sample_feature = []
    for i in negative_sample_index:
        negative_sample_feature.append(negative_feature[i])
    label1 = np.ones((len(positive_feature), 1))
    label0 = np.zeros((len(negative_sample_feature), 1))
    label = np.vstack((label1, label0))
    sample = np.vstack((positive_feature, negative_sample_feature))
    return sample, label               

sample, label = Splicing_data(interaction)   
cancer = r'/home/jby2/XH/CellMsg/dataset1'  # File directory for cancer species

# -----------------------------------------------end of data process-----------------------------------------------

class GCN_MLP(torch.nn.Module):
    def __init__(self):
        super(GCN_MLP, self).__init__()
        self.conv1 = GCNConv(2813, 1024)
        self.conv2 = GCNConv(1024, 400)
        #self.conv3 = GCNConv(400, 400)
        self.fc1 = nn.Linear(800, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)
        self.f1 = nn.Linear(2813, 1024)
        self.f2 = nn.Linear(1024, 400)

    def forward(self, x, a, sample_train):    
        r1 = self.f1(x)    
        x = self.conv1(x, a)
        x = r1 + x
        x = F.relu(x)
        
        r2 = self.f2(x)
        x = self.conv2(x, a)  
        x = x + r2
        x = F.relu(x)
        
        #r3 = x
        #x = self.conv3(x, a)
        #x = x + r3
        #x = f.relu(x)
        
        
        l_embedding = x[:interaction.shape[0], :]
        r_embedding = x[interaction.shape[0]:, :]
        feature_train = []
        
        for i in sample_train:
            l_temp = l_embedding[i[0]]
            r_temp = r_embedding[i[1]]
            temp = np.append(l_temp.detach().numpy(), r_temp.detach().numpy())
            feature_train.append(temp)
        
        x = torch.tensor(feature_train, dtype=torch.float32)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x


acc_kt = 0
precision_kt = 0
recall_kt = 0
f1_kt = 0
AUC_kt = 0
AUPR_kt = 0
mcc_kt = 0
mcc_list = []
recall_list = []

kf = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
for fold, (train_index, test_index) in enumerate(kf.split(sample, label)):
    print(f"Fold: {fold + 1}")
    feature_train, feature_test = sample[train_index], sample[test_index]
    label_train, label_test = label[train_index], label[test_index]
    #feature_train = torch.tensor(feature_train, dtype=torch.float32)
    #feature_test = torch.tensor(feature_test, dtype=torch.float32)
    label_train = torch.tensor(label_train, dtype=torch.float32)
    label_test = torch.tensor(label_test, dtype=torch.float32)
    #print(type(label_train))
    #print(type(feature_train))
    
    model = GCN_MLP().to(device)
    # lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=9*1e-4, weight_decay=1e-7)
    criterion = torch.nn.BCELoss() 
    l_emb = []
    r_emb = []
    # 250
    for epoch in range(250):  
        model.train()
        optimizer.zero_grad()
        output = model(L_R_fea.to(device), edge_index.to(device), feature_train.to(device))
        loss = criterion(output, label_train.to(device))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    model.eval()
    correct = 0
    with torch.no_grad():
        output = model(L_R_fea.to(device), edge_index.to(device), feature_test.to(device))
        #_, predicted = torch.max(output.data, 1)
        y_ = []
        list_out = output.view(-1).tolist()
        for i in list_out:
            if i < 0.55:
                y_.append(0)
            else:
                y_.append(1)
        # compute mcc
        mcc_score = matthews_corrcoef(label_test.numpy(), np.array(y_).reshape(-1, 1))
        mcc_kt += mcc_score
        print(f"Matthews Correlation Coefficient: {mcc_score}")
        acc_kt += accuracy_score(label_test.numpy(), np.array(y_).reshape(-1, 1))
        print(f"acc: {accuracy_score(label_test.numpy(), np.array(y_).reshape(-1, 1))}")
        precision_kt += precision_score(label_test.numpy(), np.array(y_).reshape(-1, 1))
        print(f"precision: {precision_score(label_test.numpy(), np.array(y_).reshape(-1, 1))}")
        recall_kt += recall_score(label_test.numpy(), np.array(y_).reshape(-1, 1))
        print(f"recall: {recall_score(label_test.numpy(), np.array(y_).reshape(-1, 1))}")
        f1_kt += f1_score(label_test.numpy(), np.array(y_).reshape(-1, 1))
        print(f"f1_score: {f1_score(label_test.numpy(), np.array(y_).reshape(-1, 1))}")
        fpr_kt, tpr_kt, thresholds = roc_curve(label_test.numpy(), output.detach().cpu().numpy())
        AUC_kt += auc(fpr_kt, tpr_kt)
        roc_auc = auc(fpr_kt, tpr_kt)
        print(f"roc_auc: {roc_auc}")
        # print(f"fpr: {fpr_kt}")
        prec_kt, rec_kt, thr = precision_recall_curve(label_test.numpy(), output.detach().cpu().numpy())
        AUPR_kt += auc(rec_kt, prec_kt)
        aupr = auc(rec_kt, prec_kt)
        print(f"aupr: {aupr}")
        
        
        with open("/home/jby2/XH/CellMsg/dataset1/five-fold-cross-validation/value_of_" + str(fold) + ".txt", mode="a") as f:
            f.write("precision:{}\n".format(precision_score(label_test.numpy(), np.array(y_).reshape(-1, 1))))
            f.write("recall:{}\n".format(recall_score(label_test.numpy(), np.array(y_).reshape(-1, 1))))
            f.write("acc:{}\n".format(accuracy_score(label_test.numpy(), np.array(y_).reshape(-1, 1))))
            f.write("f1:{}\n".format(f1_score(label_test.numpy(), np.array(y_).reshape(-1, 1))))
            f.write("AUC:{}\n".format(roc_auc))
            f.write("AUPR:{}\n".format(aupr))
            f.write("Matthews Correlation Coefficient:{}\n".format(mcc_score))
        
        
        # save fpr and tpr of every validation
        data = {'FPR': fpr_kt, 'TPR': tpr_kt}
        df = pd.DataFrame(data)
        directory = '/home/jby2/XH/CellMsg/dataset1/five-fold-cross-validation/roc/'
        #if not os.path.exists(directory):
            # create directory
            # os.makedirs(directory)
        df.to_csv(directory + str(fold) + ".csv", index=False)
        
        # save recall and precision of every validation
        data = {'Recall': rec_kt, 'Precision': prec_kt}
        df = pd.DataFrame(data)
        directory = '/home/jby2/XH/CellMsg/dataset1/five-fold-cross-validation/prc/'
        #if not os.path.exists(directory):
            # create directory
            #os.makedirs(directory)
        df.to_csv(directory + str(fold) + ".csv", index=False)
        
        mcc_list.append(mcc_score)
        recall_list.append(recall_score(label_test.numpy(), np.array(y_).reshape(-1, 1)))
        
        
        
        # save model
        #torch.save(model.state_dict(), '/home/jby2/XH/CellMsg/dataset1/test_model' + str(fold) + '.pth')
        
        # -----------------------------------------------------------------------------

acc_kt = acc_kt / 5
precision_kt = precision_kt / 5
recall_kt = recall_kt / 5
f1_kt = f1_kt / 5
AUC_kt = AUC_kt / 5
AUPR_kt = AUPR_kt / 5
mcc_kt = mcc_kt / 5

std_mcc = np.std(mcc_list, ddof=1)  
mcc_low = mcc_kt - std_mcc
mcc_high = mcc_kt + std_mcc  # mcc score between mcc_low and mcc_high is recognized similar so that to compare recall

default_model = 0
for i in range(len(mcc_list)):
    if i == 0:
        continue
    if mcc_list[i] >= mcc_low and mcc_list[i] <= mcc_high:
        if recall_list[i] > recall_list[default_model]:
            default_model = i
            

# the final model that we choose
#os.rename('/home/jby2/XH/CellMsg/dataset1/test_model' + str(default_model) + '.pth', '/home/jby2/XH/CellMsg/dataset1/final_model.pth')

print(f"mean_acc: {acc_kt}")
print(f"mean_precision: {precision_kt}")
print(f"mean_recall: {recall_kt}")
print(f"mean_auc: {AUC_kt}")
print(f"mean_aupr: {AUPR_kt}")
print(f"mean_Matthews Correlation Coefficient: {mcc_kt}")

with open("/home/jby2/XH/CellMsg/dataset1/five-fold-cross-validation/mean_value_of_five-fold_cross-validation.txt", mode="a") as f:
    f.write("precision:{}\n".format(precision_kt))
    f.write("recall:{}\n".format(recall_kt))
    f.write("acc:{}\n".format(acc_kt))
    f.write("f1:{}\n".format(f1_kt))
    f.write("AUC:{}\n".format(AUC_kt))
    f.write("AUPR:{}\n".format(AUPR_kt))
    f.write("Matthews Correlation Coefficient:{}\n".format(mcc_kt))


# plot fig of the five-fold cross-validation
CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'
colorlist = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber, CB91_Purple, CB91_Violet]

# roc
mean_fpr = np.linspace(0, 1, 1000)
tprs = []
for fold in range(5):
    data = pd.read_csv(cancer + '/five-fold-cross-validation/roc/' + str(fold) + ".csv")
    fpr = data['FPR'].tolist()
    tpr = data['TPR'].tolist()
    roc_auc = auc(fpr, tpr)
    tprs.append(interp(mean_fpr, fpr, tpr))
    plt.plot(fpr, tpr, lw=1.5, alpha=0.8, color=colorlist[fold], label='%dst fold (AUC = %0.4f)' % (fold + 1, roc_auc))
    
# plot mean data of five-fold cross-validation
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color=colorlist[5], label=r'Average (AUC = %0.4f)' % (mean_auc), lw=2, alpha=1)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate',fontsize=13)
plt.ylabel('True Positive Rate',fontsize=13)
plt.title('ROC CURVE')
plt.legend(loc='lower right')
plt.savefig(cancer + '/five-fold-cross-validation/roc/ROC-5fold.pdf',dpi=1080)
plt.close()


# pr curve
mean_precision = np.linspace(0, 1, 1000)
recalls = []
for fold in range(5):
    data = pd.read_csv(cancer + '/five-fold-cross-validation/prc/' + str(fold) + ".csv")
    precision = data['Precision'].tolist()
    recall = data['Recall'].tolist()
    aupr = auc(recall, precision)
    recalls.append(interp(mean_precision, precision, recall))
    plt.plot(recall, precision, lw=1.5, alpha=0.8, color=colorlist[fold], label='%dst fold (AUPR = %0.4f)' % (fold + 1, aupr))
    
# plot mean data of five-fold cross-validation
mean_recall = np.mean(recalls, axis=0)
mean_aupr = auc(mean_recall, mean_precision)
plt.plot(mean_recall, mean_precision, color=colorlist[5], label=r'Average (AUPR = %0.4f)' % (mean_aupr), lw=2, alpha=1)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall',fontsize=13)
plt.ylabel('Precision',fontsize=13)
plt.title('PR CURVE')
plt.legend(loc='lower left')
plt.savefig(cancer + '/five-fold-cross-validation/prc/aupr-5fold.pdf',dpi=1080)
plt.close()












