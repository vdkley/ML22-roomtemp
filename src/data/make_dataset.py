# from __future__ import annotations

# import shutil
from pathlib import Path

# import gin
# import numpy as np
import pandas as pd

# import requests
# import tensorflow as tf
# import torch
from loguru import logger

# from typing import List, Tuple

# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor

# from src.data import data_tools
# from src.data.data_tools import PaddedDatagenerator, TSDataset

# Tensor = torch.Tensor


def get_roomtemp(datadir: Path) -> pd.DataFrame:
    """loads room temperatures data, selects date-time and temperature per hour"""
    file = datadir / "MLTempDataset.csv"
    if file.exists():
        logger.info(f"Found {file}, load from disk")
        data = pd.read_csv(file)
    else:
        logger.error(f"{file} does not exist")
        raise (FileNotFoundError)

    return data
