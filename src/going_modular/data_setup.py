import torch
import torchvision
from torch import nn
from torchvision import models, datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

import matplotlib.pyplot as plt
import numpy as np

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from timeit import default_timer as timer
import os
from pathlib import Path
import zipfile
import requests
from datetime import datetime
from PIL import Image
import shutil
import random


default_train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                               transforms.RandomRotation(30),
                                               transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                               transforms.GaussianBlur(3),
                                               transforms.ToTensor()])


def create_auto_transforms(model: torch.nn.Module) -> List[transforms.Compose]:
  '''
  '''
  print("[INFO] create_auto_transforms function incomplete")


def create_dataloaders(train_path: str,
                       val_path: str,
                       test_path: str,
                       batch_size: int,
                       transforms: list,
                       num_workers: int=os.cpu_count()) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
  '''
  This function creates dataloaders for training, validation, and testing datasets.

  Args:
    train_path (str): The path to the training dataset
    val_path (str): The path to the validation dataset
    test_path (str): The path to the testing dataset
    batch_size (int): The batch size for the dataloaders
    transforms (list): A list of transforms to apply to the datasets
    num_workers (int): The number of workers to use for creating dataloaders

  Returns:
    train_dataloader (DataLoader): The training dataloader
    val_dataloader (DataLoader): The validation dataloader
    test_dataloader (DataLoader): The testing dataloader
    class_names (List[str]): A list of class names
  '''
  train_data = ImageFolder(train_path, transform=transforms[0])
  val_data = ImageFolder(val_path, transform=transforms[1])
  test_data = ImageFolder(test_path, transform=transforms[1])

  class_names = train_data.classes

  train_dataloader = DataLoader(dataset=train_data,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=True,
                                pin_memory=True)
  
  val_dataloader = DataLoader(dataset=val_data,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=False,
                              pin_memory=True)
  
  test_dataloader = DataLoader(dataset=test_data,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False,
                                pin_memory=True)

  return train_dataloader, val_dataloader, test_dataloader, class_names
