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

from data_setup import create_auto_transforms

def set_seeds(seed: int=42) -> None:
  '''
  This function sets manual seeds for reproducibility.

  Args:
    seed (int): The seed value to use

  Returns:
    None
  '''
  torch.manual_seed(seed)
  torch.cuda.manual_seed(42)
  random.seed(42)


def save_model(model: torch.nn.Module, target_dir: str, model_name: str) -> None:
  '''
  This function saves a model to a specified directory.

  Args:
    model (torch.nn.Module): The model to save
    target_dir (str): The directory to save the model
    model_name (str): The name to save the model as

  Returns:
    None
  '''
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True, exist_ok=True)

  assert model_name.endswith('.pt') or model_name.endswith(".pth"), "[ERROR] Model name must end with .pt or .pth"
  model_save_path = target_dir_path / model_name

  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(), f=model_save_path)


def eval_model(model: torch.nn.Module,
               test_dataloader: DataLoader,
               loss_fn: torch.nn.Module,
               device: torch.device) -> Tuple[float, float]:
  '''
  This function evaluates a model on a test dataset.

  Args:
    model (torch.nn.Module): The model to evaluate
    test_dataloader (DataLoader): The dataloader for the test dataset
    loss_fn (torch.nn.Module): The loss function to use
    device (torch.device): The device to use for evaluation

  Returns:
    test_loss (float): The loss on the test dataset
    test_acc (float): The accuracy on the test dataset
  '''
  test_loss, test_acc = 0, 0

  model.eval()
  with torch.inference_mode():
    for batch, (X, y) in enumerate(test_dataloader):
      X, y = X.to(device), y.to(device)

      test_logits = model(X)
      loss = loss_fn(test_logits, y)

      test_loss += loss.item()

      test_labels = torch.argmax(test_logits, dim=1)
      test_acc += ((test_labels == y).sum().item()/len(test_labels))

  test_loss /= len(test_dataloader)
  test_acc /= len(test_dataloader)

  return test_loss, test_acc


def custom_eval(model: torch.nn.Module,
                model_name: str,
                image_path: str,
                class_names: List[str],
                device: torch.device) -> Tuple[str, float]:
  '''
  This function predicts on a custom image.

  Args:
    model (torch.nn.Module): The model to use for prediction
    model_name (str): The name of the model
    image_path (str): The path to the image
    class_names (list): The class names for the model
    device (torch.device): The device to use for prediction

  Returns:
    pred_class (str): The predicted class
    pred_prob (float): The predicted probability
  '''

  auto_transforms = create_auto_transforms(model_name)

  image = Image.open(image_path)
  transformed_image = auto_transforms(image).unsqueeze(dim=1).to(device)

  model.to(device)
  model.eval()

  with torch.inference_mode():
    pred_logits = model(transformed_image)
    pred_probs = torch.softmax(pred_logits, dim=1)
    pred_label = pred_logits.argmax(dim=1)

    pred_class = class_names[pred_label]
    pred_prob = pred_probs.max()
  
  return pred_class, pred_prob
