import numpy as np
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from transformers import Trainer, TrainingArguments
import pandas as pd
import torch

print("AUDIO and TEXT parameters")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load the CSV files into DataFrames
df1 = pd.read_csv('all_text_features.csv')
df2 = pd.read_csv('all_audio_features.csv')
df3 = pd.read_csv('all_video_features.csv')

# Merge the DataFrames on 'row_id' and 'clip_id'
text_audio_merged_df = pd.merge(df1, df2, on=['video_id', 'clip_id', 'mode', 'annotation_label'], how='inner')
audio_video_merged_df = pd.merge(df3, df2, on=['video_id', 'clip_id', 'mode', 'annotation_label'], how='inner')
text_video_merged_df = pd.merge(df3, df1, on=['video_id', 'clip_id', 'mode', 'annotation_label'], how='inner')
all_merged_df = pd.merge(text_audio_merged_df, df3, on=['video_id', 'clip_id', 'mode', 'annotation_label'], how='inner')

# Filter the DataFrame by mode (train, valid, test) to create subsets
train_all_merged_df = all_merged_df[all_merged_df["mode"] == "train"].drop(columns=["mode"])
train_text_audio_merged_df = text_audio_merged_df[text_audio_merged_df["mode"] == "train"].drop(columns=["mode"])
train_audio_video_merged_df = audio_video_merged_df[audio_video_merged_df["mode"] == "train"].drop(columns=["mode"])
train_text_video_merged_df = text_video_merged_df[text_video_merged_df["mode"] == "train"].drop(columns=["mode"])

valid_all_merged_df = all_merged_df[all_merged_df["mode"] == "valid"].drop(columns=["mode"])
valid_text_audio_merged_df = text_audio_merged_df[text_audio_merged_df["mode"] == "valid"].drop(columns=["mode"])
valid_audio_video_merged_df = audio_video_merged_df[audio_video_merged_df["mode"] == "valid"].drop(columns=["mode"])
valid_text_video_merged_df = text_video_merged_df[text_video_merged_df["mode"] == "valid"].drop(columns=["mode"])

test_all_merged_df = all_merged_df[all_merged_df["mode"] == "test"].drop(columns=["mode"])
test_text_audio_merged_df = text_audio_merged_df[text_audio_merged_df["mode"] == "test"].drop(columns=["mode"])
test_audio_video_merged_df = audio_video_merged_df[audio_video_merged_df["mode"] == "test"].drop(columns=["mode"])
test_text_video_merged_df = text_video_merged_df[text_video_merged_df["mode"] == "test"].drop(columns=["mode"])

class MLP_Model(nn.Module):
    def __init__(self, layer_sizes, output_dim=2, dropout_p=0, act_func="tanh"):
        super(MLP_Model, self).__init__()
        self.layer_sizes = layer_sizes
        self.output_dim = output_dim
        self.dropout_prob = dropout_p

        # Create a list to hold all layers
        self.layers = nn.ModuleList()

        # Input layer normalization
        # self.layer_norm1 = nn.LayerNorm(layer_sizes[0])

        # Add the first layer
        self.layers.append(nn.Linear(layer_sizes[0], layer_sizes[1]))

        # Add intermediate layers
        for i in range(1, len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

        # Output layer
        self.out = nn.Linear(layer_sizes[-1], output_dim)

        # Dropout and activation
        self.dropout = nn.Dropout(dropout_p)
        if act_func == "tanh":
            self.act = nn.Tanh()
        elif act_func == "relu":
            self.act = nn.ReLU()

    def forward(self, x, labels=None):
        # Apply LayerNorm to the input
        # x = self.layer_norm1(x)

        # Pass through all layers
        for layer in self.layers:
            x = layer(x)
            x = self.act(x)
            x = self.dropout(x)

        # Output layer
        x = self.out(x)
        return x
    
from torch.utils.data import DataLoader

def collate_fn(batch):
    # Convert lists to tensors
    input_ids = torch.tensor([item['input_ids'] for item in batch], dtype=torch.float32)
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    
    return {'input_ids': input_ids, 
            'labels': labels
           }


# ===================================================================================
## TEXT AND AUDIO
# ===================================================================================
train_text_audio_merged_labels = train_text_audio_merged_df['annotation_label']
train_text_audio_merged_features = train_text_audio_merged_df.drop(columns=['video_id', 'clip_id', 'annotation_label'])
train_text_audio_merged_features = train_text_audio_merged_features.values.tolist()
# Create datasets with the features and labels
train_text_audio_merged_dataset = Dataset.from_dict({
    'input_ids': train_text_audio_merged_features,
    'labels': train_text_audio_merged_labels.tolist()
})

valid_text_audio_merged_labels = valid_text_audio_merged_df['annotation_label']
valid_text_audio_merged_features = valid_text_audio_merged_df.drop(columns=['video_id', 'clip_id', 'annotation_label'])
valid_text_audio_merged_features = valid_text_audio_merged_features.values.tolist()
# Create datasets with the features and labels
valid_text_audio_merged_dataset = Dataset.from_dict({
    'input_ids': valid_text_audio_merged_features,
    'labels': valid_text_audio_merged_labels.tolist()
})

test_text_audio_merged_labels = test_text_audio_merged_df['annotation_label']
test_text_audio_merged_features = test_text_audio_merged_df.drop(columns=['video_id', 'clip_id', 'annotation_label'])
test_text_audio_merged_features = test_text_audio_merged_features.values.tolist()
# Create datasets with the features and labels
test_text_audio_merged_dataset = Dataset.from_dict({
    'input_ids': test_text_audio_merged_features,
    'labels': test_text_audio_merged_labels.tolist()
})

# Create the DatasetDict to hold the subsets
dataset_text_audio_merged = DatasetDict({
    'train': train_text_audio_merged_dataset,
    'valid': valid_text_audio_merged_dataset,
    'test': test_text_audio_merged_dataset
})

def train(model, train_loader, criterion, optimizer):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []  # To store all predictions
    all_labels = []  # To store all ground truth labels

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device).float()
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Store predictions and labels
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Calculate average loss and accuracy
    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total

    return avg_loss, accuracy, all_preds, all_labels

def evaluate(model, data_loader, criterion, optimizer):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []  # To store all predictions
    all_labels = []  # To store all ground truth labels

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device).float()
            labels = batch['labels'].to(device)

            outputs = model(input_ids)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate average loss and accuracy
    avg_loss = running_loss / len(data_loader)
    accuracy = correct / total

    return avg_loss, accuracy, all_preds, all_labels

import itertools
import torch.optim as optim
def grid_search(layer_size_options, dropout_options, act_func_options, batch_options, lr_options, weightdecay_options,
                train_set, valid_set, 
                device,
                max_epochs=15):
    best_model = None
    best_valid_accuracy = 0.0
    best_params = None
    best_epoch = 0
    
    param_combinations = list(itertools.product(layer_size_options, dropout_options, act_func_options, batch_options, lr_options, weightdecay_options))
    
    model_versions = []  # Store the best model for each hyperparameter combination
    
    for layer_sizes, dropout_p, act_func, batch_size, lr, wd in param_combinations:
        print(f"Testing configuration: layer_sizes={layer_sizes}, dropout_p={dropout_p}, act_func={act_func}, batch_size={batch_size} , lr={lr}, wd={wd}")

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False, collate_fn=collate_fn)
        
        # Initialize the model
        model = MLP_Model(layer_sizes=layer_sizes, dropout_p=dropout_p, act_func=act_func)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        
        # Define optimizer
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        
        best_loss_for_config = float('inf')
        best_accuracy_for_config = 0.0
        best_epoch_for_config = 0
        best_model_for_config = None
        
        for epoch in range(max_epochs):
            # print(f'Epoch {epoch + 1} / {max_epochs}')
            train_loss, _, _, _ = train(model, train_loader, criterion, optimizer)
            valid_loss, valid_accuracy, _, _ = evaluate(model, valid_loader, criterion, optimizer)
            
            # print(f'Validation Loss: {valid_loss:.3f}, Validation Accuracy: {valid_accuracy:.3f}')
            
            if valid_loss < best_loss_for_config:
                best_loss_for_config = valid_loss
                best_accuracy_for_config = valid_accuracy
                best_epoch_for_config = epoch + 1
                best_model_for_config = model
                # torch.save(model.state_dict(), f'model_config_{layer_sizes}_{dropout_p}_{act_func}_best.bin')
        
        print(f"Best epochs: {best_epoch_for_config}, Min. Loss: {best_loss_for_config}, Accuracy: {best_accuracy_for_config}")
        model_versions.append((best_accuracy_for_config, best_loss_for_config, best_model_for_config, layer_sizes, dropout_p, act_func, best_epoch_for_config))
    
    # Select the best model: highest accuracy among the best-loss models
    model_versions.sort(key=lambda x: -x[0])  # Sort by accuracy (descending)
    best_valid_accuracy, best_valid_loss, best_model, best_layer_sizes, best_dropout_p, best_act_func, best_epoch = model_versions[0]
    
    print(f'Best Configuration: layer_sizes={best_layer_sizes}, dropout_p={best_dropout_p}, act_func={best_act_func}')
    print(f'Best Validation Loss: {best_valid_loss:.3f}, Best Validation Accuracy: {best_valid_accuracy:.3f}, Best Epoch: {best_epoch}')
    
    return best_model, (best_layer_sizes, best_dropout_p, best_act_func)

# Define parameter search space
layer_sizes_options = [[1024, 512], 
                       [1024, 256], 
                       [1024, 128], 
                       [1024, 32],
                       [1024, 1024, 256], 
                       [1024, 1024, 128], 
                       [1024, 1024, 32],
                       [1024, 512, 256], 
                       [1024, 512, 128], 
                       [1024, 512, 32],
                       [1024, 1024, 512, 128], 
                       [1024, 1024, 512, 32],
                       [1024, 512, 256, 128], 
                       [1024, 512, 256, 32], 
                       [1024, 512, 128, 32],
                       [1024, 512, 256, 128, 32]]
batch_options = [32, 64, 128]
lr_options = [5e-5, 1e-5, 5e-6]
weightdecay_options = [1e-3, 1e-4, 1e-5]
dropout_options = [0.1, 0.3]
act_func_options = ["tanh", "relu"]

# Run grid search
best_model, best_params = grid_search(
    layer_sizes_options, dropout_options, act_func_options, batch_options, lr_options, weightdecay_options,
    dataset_text_audio_merged['train'], dataset_text_audio_merged['valid'],
    device
)