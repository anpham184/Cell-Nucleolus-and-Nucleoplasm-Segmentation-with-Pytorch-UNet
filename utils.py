import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
import numpy as np
from model import UNet
import yaml

class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    @property
    def EPOCHS(self):
        return self.config.get('EPOCHS')

    @property
    def LEARNING_RATE(self):
        return self.config.get('LEARNING_RATE')

    @property
    def BATCH_SIZE(self):
        return self.config.get('BATCH_SIZE')

    @property
    def NUM_CLASSES(self):
        return self.config.get('NUM_CLASSES')

    @property
    def IN_CHANNELS(self):
        return self.config.get('IN_CHANNELS')

    @property
    def WORKING_DIR(self):
        return self.config.get('WORKING_DIR')

    @property
    def SAVE_DIR(self):
        return self.config.get('SAVE_DIR')

    @property
    def SAVE_CHECKPOINTS(self):
        return self.config.get('SAVE_CHECKPOINTS')

    @property
    def DEVICE(self):
        return self.config.get('DEVICE')

    @property
    def NUM_WORKERS(self):
        return self.config.get('NUM_WORKERS')

    @property
    def SAVE_PLOT(self):
        return self.config.get('SAVE_PLOT')

    @property
    def MODEL_PATH(self):
        return self.config.get('MODEL_PATH')

    @property
    def SAVE_PREDICT(self):
        return self.config.get('SAVE_PREDICT')


# Define the dataset
class TrainDataset(Dataset):
    def __init__(self, root_path, limit=None, num_classes=3):
        self.root_path = root_path
        self.limit = limit
        self.num_classes = num_classes
        self.images = sorted([os.path.join(root_path, "train", i) for i in os.listdir(os.path.join(root_path, "train"))])[:self.limit]
        self.masks = sorted([os.path.join(root_path, "train_masks", i) for i in os.listdir(os.path.join(root_path, "train_masks"))])[:self.limit]

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()])

        if self.limit is None:
            self.limit = len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")

        transformed_img = self.transform(img)
        transformed_mask = self.transform_mask(mask)
        #transformed_mask = self.transform(mask)

        img_name = os.path.basename(self.images[index])

        return transformed_img, transformed_mask, img_name
       
    def transform_mask(self, mask):
        mask = mask.resize((512, 512), Image.NEAREST)
        mask_tensor = torch.zeros((self.num_classes, 512, 512))

        # Initialize a tensor to hold the one-hot encoded mask
        for class_index in range(self.num_classes):
            mask_class = torch.tensor(np.array(mask) == class_index)
            mask_tensor[class_index] = mask_class
    
        return mask_tensor
    
    def __len__(self):
        return min(len(self.images), self.limit)
 
class TestDataset(Dataset):
    def __init__(self, root_path, limit=None, num_classes=3):
        self.root_path = root_path
        self.limit = limit
        self.num_classes = num_classes
        self.images = sorted([os.path.join(root_path, "train", i) for i in os.listdir(os.path.join(root_path, "train"))])[:self.limit]
        self.masks = sorted([os.path.join(root_path, "train_masks", i) for i in os.listdir(os.path.join(root_path, "train_masks"))])[:self.limit]

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()])

        if self.limit is None:
            self.limit = len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")

        transformed_img = self.transform(img)
        transformed_mask = self.transform_mask(mask)
        #transformed_mask = self.transform(mask)

        img_name = os.path.basename(self.images[index])

        return transformed_img, transformed_mask, img_name
    
    def transform_mask(self, mask):
        mask = mask.resize((512, 512), Image.NEAREST)
        mask_tensor = torch.zeros((self.num_classes, 512, 512))

        # Initialize a tensor to hold the one-hot encoded mask
        for class_index in range(self.num_classes):
            mask_class = torch.tensor(np.array(mask) == class_index)
            mask_tensor[class_index] = mask_class

        return mask_tensor
    
    def __len__(self):
        return min(len(self.images), self.limit)


class AugmentedDataset(Dataset):
    def __init__(self, base_dataset, num_augmentations=10):
        self.base_dataset = base_dataset
        self.num_augmentations = num_augmentations
        self.augmentation_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(512, scale=(0.8, 1.0))
        ])

    def __getitem__(self, index):
        original_index = index % len(self.base_dataset)
        img, mask, _ = self.base_dataset[original_index]
        
        if index >= len(self.base_dataset):
            seed = torch.randint(0, 2**32, (1,), dtype=torch.int64).item()
            torch.manual_seed(seed)
            img = self.augmentation_transforms(img)
            
            torch.manual_seed(seed)
            mask = self.augmentation_transforms(mask)
        
        return img, mask

    def __len__(self):
        return len(self.base_dataset) * self.num_augmentations

'''
def dice_coefficient(prediction, target, epsilon=1e-07):
    prediction_copy = prediction.clone()
    prediction_copy[prediction_copy < 0] = 0
    prediction_copy[prediction_copy > 0] = 1
    intersection = abs(torch.sum(prediction_copy * target))
    union = abs(torch.sum(prediction_copy) + torch.sum(target))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    
    return dice
'''
class UNetInferenceSaver:
    def __init__(self, model_pth, device, save_dir):
        self.device = device
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.model = UNet(in_channels=3, num_classes=3).to(self.device)
        self.model.load_state_dict(torch.load(model_pth, map_location=torch.device(self.device)))
        self.model.eval()  # Set model to evaluation mode

        self.transform = transforms.Compose([
            transforms.Resize((512, 512))
        ])

    def save_predictions(self, test_dataloader):
        for batch_idx, (img_tensor_batch, _, img_name_batch) in enumerate(test_dataloader):
            for img_tensor, img_name in zip(img_tensor_batch, img_name_batch):
                img = img_tensor.unsqueeze(0).to(self.device)  # Add batch dimension and move to device

                with torch.no_grad():
                    pred_mask = self.model(img)

                pred_mask = pred_mask.squeeze(0)  # Remove batch dimension
                pred_mask = torch.softmax(pred_mask, dim=0)  # Apply softmax to get probabilities

                # Convert the predicted mask to a numpy array
                pred_mask_np = pred_mask.cpu().numpy()

                # Get the predicted class for each pixel
                pred_class = np.argmax(pred_mask_np, axis=0)

                # Convert the predicted class array to an image
                pred_mask_img = Image.fromarray(pred_class.astype(np.uint8))

                # Get filename without extension
                img_name_only, _ = os.path.splitext(img_name)

                # Save the predicted mask
                pred_mask_img.save(os.path.join(self.save_dir, f"pred_{img_name_only}.tif"))


def dice_coefficient_multiclass(prediction, target, num_classes, epsilon=1e-07):
    dice_scores = []
    
    def binarize_prediction(prediction_class, threshold=0.5):
        return (prediction_class >= threshold).float()

    # Iterate over each class
    for class_index in range(num_classes):
        # Extract the channel corresponding to the current class
        prediction_class = prediction[:, class_index, :, :]
        target_class = target[:, class_index, :, :]

        # Apply sigmoid activation
        prediction_class = torch.sigmoid(prediction_class)
        
        # Binarize the prediction_class using the threshold
        prediction_class = binarize_prediction(prediction_class)
        
        # Binarize the target_class if needed (assuming target channels are one-hot encoded)
        target_class = (target_class > 0.5).float()
        
        # Calculate intersection and union per batch item
        intersection = torch.sum(prediction_class * target_class)
        union = torch.sum(prediction_class) + torch.sum(target_class)
        
        # Calculate Dice coefficient for the current class per batch item
        dice = (2. * intersection + epsilon) / (union + epsilon)
        dice_scores.append(dice.mean())
    
    # Stack dice scores for all classes and calculate the mean Dice coefficient across all classes
    mean_dice = torch.mean(torch.stack(dice_scores))
    
    return mean_dice
# Function to plot training results
def plot_training_results(epochs, train_losses, val_losses, train_dcs, val_dcs, save_plot):
    epochs_list = list(range(1, epochs + 1))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, train_losses, label='Training Loss')
    plt.plot(epochs_list, val_losses, label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, train_dcs, label='Training DICE')
    plt.plot(epochs_list, val_dcs, label='Validation DICE')
    plt.title('DICE Coefficient over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('DICE')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_plot + "train_loss_dice.jpg", dpi=300)
    plt.show()






    

