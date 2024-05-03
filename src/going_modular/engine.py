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


def train_step(model: torch.nn.Module,
               train_dataloader: DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
  '''
  This function performs one training step on a batch of data.

  Args:
    model (torch.nn.Module): The model to train
    train_dataloader (DataLoader): The dataloader for the training dataset
    loss_fn (torch.nn.Module): The loss function to use
    optimizer (torch.optim.Optimizer): The optimizer to use
    device (torch.device): The device to use for training

  Returns:
    train_loss (float): The loss on the training dataset
    train_acc (float): The accuracy on the training dataset
  '''
  model.train()
  train_loss, train_acc = 0, 0

  for batch, (X, y) in enumerate(train_dataloader):
    X, y = X.to(device), y.to(device)

    train_logits = model(X)
    train_labels = train_logits.argmax(dim=1)

    loss = loss_fn(train_logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    train_acc += ((train_labels == y).sum().item()/len(train_labels))
  
  train_loss /= len(train_dataloader)
  train_acc /= len(train_dataloader)

  return train_loss, train_acc


def val_step(model: torch.nn.Module,
             val_dataloader: DataLoader,
             loss_fn: torch.nn.Module,
             device: torch.device) -> Tuple[float, float]:
  '''
  This function performs one validation step on a batch of data.

  Args:
    model (torch.nn.Module): The model to validate
    val_dataloader (DataLoader): The dataloader for the validation dataset
    loss_fn (torch.nn.Module): The loss function to use
    device (torch.device): The device to use for validation
  '''
  model.eval()
  val_loss, val_acc = 0, 0

  with torch.inference_mode():
    for batch, (X, y) in enumerate(val_dataloader):
      X, y = X.to(device), y.to(device)

      val_logits = model(X)
      val_labels = val_logits.argmax(dim=1)

      loss = loss_fn(val_logits, y)

      val_loss += loss.item()
      val_acc += ((val_labels == y).sum().item()/len(val_labels))
    
  val_loss /= len(val_dataloader)
  val_acc /= len(val_dataloader)

  return val_loss, val_acc


def train(model: torch.nn.Module,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          scheduler: torch.optim.lr_scheduler,
          num_epochs: int,
          patience: int=5,
          minDelta: float=0.01) -> Dict[str, List[float]]:
  '''
  This function trains a model on a dataset.

  Args:
    model (torch.nn.Module): The model to train
    train_dataloader (DataLoader): The dataloader for the training dataset
    val_dataloader (DataLoader): The dataloader for the validation dataset
    loss_fn (torch.nn.Module): The loss function to use
    optimizer (torch.optim.Optimizer): The optimizer to use
    device (torch.device): The device to use for training
    scheduler (torch.optim.lr_scheduler): The learning rate scheduler to use
    num_epochs (int): The number of epochs to train for
    patience (int): The number of epochs to wait before early stopping
    minDelta (float): The minimum delta for early stopping

  Returns:
    results (Dict[str, List[float]]): A dictionary of results, including training loss, training accuracy, validation loss, and validation accuracy
  '''
  results = {"train_loss": [],
             "train_acc": [],
             "val_loss": [],
             "val_acc": []}
  
  current_patience = 0
  best_loss = float('inf')

  for epoch in tqdm(range(num_epochs)):
    train_loss, train_acc = train_step(model=model,
                                       train_dataloader=train_dataloader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       device=device)
    
    print(f'Epoch {epoch + 1}/{num_epochs}:\nTraining Loss: {train_loss:.2f} | Training Acc: {train_acc:.2f}%')
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    
    val_loss, val_acc = val_step(model=model,
                                  val_dataloader=val_dataloader,
                                  loss_fn=loss_fn,
                                  device=device)
    
    print(f'Validation Loss: {val_loss:.2f} | Validation Acc: {val_acc:.2f}%')
    results["val_loss"].append(val_loss)
    results["val_acc"].append(val_acc)

    if val_loss < best_loss - minDelta:
      best_loss = val_loss
      current_patience = 0
    else:
      current_patience += 1
      if current_patience == patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break
    
    try:
        scheduler.step(val_loss)
    except:
        scheduler.step()
