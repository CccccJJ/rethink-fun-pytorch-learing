from torch.utils.data import DataLoader
from titanic_dataset import TitanicDataset

dataset = TitanicDataset(r"./chapter7/titanic/train.csv")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for inputs, labels in dataloader:
    print(inputs.shape, labels.shape)
    break
