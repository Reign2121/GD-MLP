import torch
from torch.utils.data import Dataset, DataLoader
#from sklearn.preprocessing import StandardScaler


class Custom_Seq_Dataset(Dataset):
    def __init__(self, data, input_len,label_len, pred_len, target):
        self.data = data
        self.target = data[target].values #넘파이
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
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(1) #unsqueeze, 차원확장 (feature 차원 추가)
        
        return feature, target
      
      data = Custom_Seq_Dataset(data=data,input_len=336, label_len=96,pred_len=96,target='OT') #Hyperparameter
    

#분할    
train_size = int(0.7 * len(data))  #분할했을 때 수 반환해서 미리 체크
val_size = int(0.1 * len(data))    
test_size = len(data) - train_size - val_size

train_indices = list(range(train_size)) #데이터를 시간의 순서대로 분할하기 위해 순서대로 인덱스 생성
val_indices = list(range(train_size, train_size + val_size))
test_indices = list(range(train_size + val_size, len(data)))

from torch.utils.data import Subset #하위 데이터로 나누기.
train_set = Subset(data, train_indices) #(데이터 셋, 인덱스).
val_set = Subset(data, val_indices)
test_set = Subset(data, test_indices)
      
      
