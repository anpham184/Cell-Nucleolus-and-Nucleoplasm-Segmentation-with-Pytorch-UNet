import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from model import UNet
from utils import Config, TrainDataset, AugmentedDataset, dice_coefficient_multiclass, plot_training_results
import matplotlib.pyplot as plt
import numpy as np
import yaml

def train():
    # Prepare dataset
    train_dataset = TrainDataset(config.WORKING_DIR)
    generator = torch.Generator().manual_seed(42)
    #augmented_train_dataset = AugmentedDataset(train_dataset, num_augmentations=10)
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset, 
                                    num_workers=config.NUM_WORKERS, 
                                    batch_size=config.BATCH_SIZE, 
                                    shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, 
                                    num_workers=config.NUM_WORKERS, 
                                    batch_size=config.BATCH_SIZE, 
                                    shuffle=True)

    # Initialize model, optimizer, and loss function
    model = UNet(in_channels=config.IN_CHANNELS, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    #criterion = nn.BCEWithLogitsLoss()  # For binary masks, use nn.CrossEntropyLoss() for multi-class
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class

    torch.cuda.empty_cache()
    train_losses = []
    train_dcs = []
    val_losses = []
    val_dcs = []

    # Training loop
    for epoch in tqdm(range(config.EPOCHS)):
        model.train()
        train_running_loss = 0
        train_running_dc = 0
        
        for idx, img_mask in enumerate(tqdm(train_dataloader, position=0, leave=True)):
            img = img_mask[0].to(config.DEVICE)
            mask = img_mask[1].to(config.DEVICE)
            y_pred = model(img)

            dc = dice_coefficient_multiclass(y_pred, mask, num_classes=config.NUM_CLASSES, epsilon=1e-07)
            loss = criterion(y_pred, mask)
            
            train_running_loss += loss.item()
            train_running_dc += dc.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)
        train_dc = train_running_dc / (idx + 1)
        
        train_losses.append(train_loss)
        train_dcs.append(train_dc)

        model.eval()
        val_running_loss = 0
        val_running_dc = 0
        
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader, position=0, leave=True)):
                img = img_mask[0].to(config.DEVICE)
                mask = img_mask[1].to(config.DEVICE)

                y_pred = model(img)
                loss = criterion(y_pred, mask)
                dc = dice_coefficient_multiclass(y_pred, mask, num_classes=config.NUM_CLASSES, epsilon=1e-07)
                
                val_running_loss += loss.item()
                val_running_dc += dc.item()

            val_loss = val_running_loss / (idx + 1)
            val_dc = val_running_dc / (idx + 1)
        
        val_losses.append(val_loss)
        val_dcs.append(val_dc)
        print("-" * 30)
        print(f"Training Loss EPOCH {epoch + 1}: {train_loss:.4f}")
        print(f"Training DICE EPOCH {epoch + 1}: {train_dc:.4f}")
        print("\n")
        print(f"Validation Loss EPOCH {epoch + 1}: {val_loss:.4f}")
        print(f"Validation DICE EPOCH {epoch + 1}: {val_dc:.4f}")
        print("-" * 30)

        # Save checkpoint after each epoch
        #torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'checkpoint_epoch_{epoch + 1}.pth'))

    # Save the final model
    torch.save(model.state_dict(), config.SAVE_CHECKPOINTS)

    # Plot Training Results
    plot_training_results(config.EPOCHS, train_losses, val_losses, train_dcs, val_dcs, config.SAVE_PLOT)

if __name__ == "__main__":

    # Load configuration from YAML file
    config = Config("config.yaml")

    # Start training process
    train()