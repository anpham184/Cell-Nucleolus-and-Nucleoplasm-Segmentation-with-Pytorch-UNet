import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from model import UNet
from utils import Config, TestDataset, UNetInferenceSaver
import os
import numpy as np
from PIL import Image
import yaml


def prediction():
    
    test_data = TestDataset(config.WORKING_DIR)
    load_testdata = DataLoader(dataset=test_data,
                                num_workers=config.NUM_WORKERS, pin_memory=False,
                                batch_size=config.BATCH_SIZE,
                                shuffle=False)

    # Create an instance of the UNetInferenceSaver class
    inference_saver = UNetInferenceSaver(config.MODEL_PATH, device=config.DEVICE, save_dir=config.SAVE_PREDICT)

    # Save predictions
    inference_saver.save_predictions(load_testdata)

if __name__ == "__main__":

    # Load configuration from YAML file
    config = Config("config.yaml")


    # Start predicting process
    prediction()