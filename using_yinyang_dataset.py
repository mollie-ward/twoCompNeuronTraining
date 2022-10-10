from dataset import YinYangDataset
from torch.utils.data import DataLoader

BATCH_SIZE = 1

yy_dataset = YinYangDataset()
dataloader = DataLoader(yy_dataset, batch_size=BATCH_SIZE, shuffle=True)

for train_features, train_labels in dataloader:
    print(train_features.numpy(), train_labels.numpy())
    print(train_features.numpy()[0][0] + train_features.numpy()[0][2], train_features.numpy()[0][1] + train_features.numpy()[0][3])
    print('hi')
