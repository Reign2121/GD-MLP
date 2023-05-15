import torch
from torch.utils.data import Dataset, DataLoader
#from sklearn.preprocessing import StandardScaler


class Custom_Seq_Dataset(Dataset):
    def __init__(self, data, input_len,label_len, pred_len, target):
        self.data = data
        self.target = data[target].values #넘파이로 넣어야 한대
        self.data = data.values
        self.input_len = input_len
        self.label_len = label_len
        self.pred_len = pred_len

        
    def __len__(self):
        return len(self.data) - self.input_len - self.label_len + 1

    def __getitem__(self, idx):
        # feature and target
        feature = self.data[idx:idx+self.input_len]
        target = self.target[idx+(self.input_len-1):(idx+self.input_len + self.label_len-1)]

        # convert to tensor and reshape
        feature = torch.tensor(feature, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        #feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0) #이렇게 하니까 차원이 하나 더 생기네...
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(1) #unsqueeze, 차원확장
        
        return feature, target
      
      data = Custom_Seq_Dataset(data=data,input_len=336, label_len=96,pred_len=96,target='OT') #Hyperparameter
      
      
