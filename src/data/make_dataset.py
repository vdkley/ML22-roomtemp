from pathlib import Path
import pandas as pd
import torch
from loguru import logger
from typing import List, Tuple
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

def get_roomtemp(datadir: Path, split: int) -> Tuple[DataLoader, DataLoader]:
    """loads room temperatures data, selects date-time and temperature per hour"""
    file = datadir / "MLTempDataset.csv"
    if file.exists():
        logger.info(f"Found {file}, load from disk")
        data = pd.read_csv(file)
    else:
        logger.error(f"{file} does not exist")
        raise (FileNotFoundError)

    data = data.rename({'DAYTON_MW': 'temp', 'Datetime1': 'hour'}, axis=1)
    data = data.drop(data.columns[[0, 3]], axis=1)

    series = data['temp']
    tensordata = torch.from_numpy(series.to_numpy()).type(torch.float32)

    # Train test split
    train = tensordata[:split]

    norm = max(train)
    test = tensordata[split:]

    train = train / norm
    test = test/ norm

    
    trainset = trainset[...,None]
    testset = testset[..., None]
    trainset.shape, testset.shape

    
    class CustomDataset(Dataset):
        def __init__(self, data, horizon):
            self.data = data
            self.size = len(data)
            self.horizon = horizon
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # get a single item
            item = self.data[idx]
            # slice off the horizon
            x = item[:-self.horizon,:]
            y = item[-self.horizon:,:].squeeze(-1) # squeeze will remove the last dimension if possible.
            return x, y

    horizon = 3
    traindataset = CustomDataset(trainset, horizon=horizon)
    testdataset = CustomDataset(testset, horizon=horizon)

    trainloader = DataLoader(traindataset, batch_size=32, shuffle=True)
    testloader = DataLoader(testdataset, batch_size=32, shuffle=True)

    return trainloader, testloader

