import torch
from torch.utils.data import Dataset, DataLoader

batch_size=8 #same as D-Linear in Exchange Rate

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

train_features, train_labels = next(iter(train_loader))
test_features, test_labels = next(iter(test_loader))

#check it
print(train_labels.shape)
print(train_features.shape)
print(test_labels.shape)
print(test_features.shape)
