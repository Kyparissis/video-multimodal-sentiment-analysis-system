import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer, AlbertTokenizer, AlbertModel, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import KFold, ParameterGrid
import json

from sklearn.metrics import f1_score
import gc

# Function to write logs
def write_log(message):
    print(message)
    with open(log_file_path, 'a') as log_file:  # Open file in append mode
        log_file.write(message + '\n')  # Write the message and add a newline

# --------------------------------------------------------------------------------------------------------
# ----------------------------------------- Names --------------------------------------------------------
# --------------------------------------------------------------------------------------------------------

# Subjob name
subjob_name = 'hidd32'
# Model name
model_name = 'ELECTRA'
# Base path
base_path = '/home/k/kyparkypar/ondemand/data/sys/myjobs/projects/default/text-modality/ELECTRA/'
# Log file path
log_file_path = base_path + model_name + '_TrainingLog_' + subjob_name + '.txt'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
write_log(f'Using device: {device}')

# --------------------------------------------------------------------------------------------------------
# ---------------------------------------- Dataset -------------------------------------------------------
# --------------------------------------------------------------------------------------------------------

df = pd.read_csv("/home/k/kyparkypar/ondemand/data/sys/myjobs/projects/default/dataset/CMU-MOSI/label_edited.csv",
                 sep='\t',
                 encoding='utf-8',
                 header=0)

# --------------------------------------------------------------------------------------------------------
# ------------------------------------- Dataset Split ----------------------------------------------------
# --------------------------------------------------------------------------------------------------------

df = df[['processed_text', 'mode', 'annotation_label']].dropna()

# MULTILABEL CLASSIFICATION
# label_mapping = {-1: 0, 0: 1, 1: 2} # Map labels {-1, 0, 1} -> {0, 1, 2}
# df['annotation_label'] = df['annotation_label'].map(label_mapping).astype(int)
# # Split data into train, validation, and test sets
# train_text, test_text, train_labels, test_labels = train_test_split(df['processed_text'], df['annotation_label'],
#                                                                     random_state=42,
#                                                                     test_size=0.1,
#                                                                     stratify=df['annotation_label'])
# OR
# BINARY CLASSIFICATION
df['annotation_label'] = df['annotation_label'].astype(float)
train_text = df[df['mode'].isin(['train'])]['processed_text']
train_labels = df[df['mode'].isin(['train'])]['annotation_label']

# --------------------------------------------------------------------------------------------------------
# --------------------------------------- Tokenizer ------------------------------------------------------
# --------------------------------------------------------------------------------------------------------

# tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
# tokenizer = AutoTokenizer.from_pretrained('roberta-base')
# tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')
tokenizer = AutoTokenizer.from_pretrained('google/electra-base-discriminator')

# --------------------------------------------------------------------------------------------------------
# ---------------------------------- Hyperparameter Grid -------------------------------------------------
# --------------------------------------------------------------------------------------------------------

param_grid = {
    'epochs': [5],
    'hidden_size': [32],
    'batch_size': [8, 16, 32],
    'dropout_p': [0.1, 0.3],
    'learning_rate': [5e-5, 1e-5, 5e-6],
    'activation_fn': ['tanh', 'relu', 'No'],
    'weight_decay': [1e-4, 5e-4]
}

MAX_LEN = 128

# --------------------------------------------------------------------------------------------------------
# ----------------------------------- Functions & Model --------------------------------------------------
# --------------------------------------------------------------------------------------------------------

def tokenize_and_encode(texts):
    return tokenizer.batch_encode_plus(
        texts.tolist(),
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False
    )

def to_tensor(encoded_dict, labels):
    return (
        torch.tensor(encoded_dict['input_ids']),
        torch.tensor(encoded_dict['attention_mask']),
        torch.tensor(labels.tolist())
    )

def create_data_loader(seq, mask, labels, sampler, batch_size):
    data = TensorDataset(seq, mask, labels)
    return DataLoader(data, sampler=sampler, batch_size=batch_size)

# Define the model architecture
class ELECTRA_Arch(nn.Module):
    def __init__(self, electra, dropout_p=0.1, hidden_size=768, activation_fn='No'):
        super(ELECTRA_Arch, self).__init__()
        # ELECTRA model
        self.electra = electra
        # Dropout layer
        self.dropout = nn.Dropout(dropout_p)
        # Activation Function
        if activation_fn == 'relu':
            self.actvfunc = nn.ReLU()
        elif activation_fn == 'tanh':
            self.actvfunc = nn.Tanh()
        elif activation_fn == 'No':
            self.actvfunc = nn.Identity()
        else:
            raise ValueError("Invalid activation function")
        # FC Layer
        self.fc = nn.Linear(768, hidden_size)   # ELECTRA outputs 768 by default
        # Output layer
        # MULTILABEL CLASSIFICATION
        # self.outp = nn.Linear(hidden_size, 3)    # Assuming 3 classes for classification
        # self.logsoftmax = nn.LogSoftmax(dim=1)
        # BINARY CLASSIFICATION
        self.outp = nn.Linear(hidden_size, 1)    # Assuming 2 classes for classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, sent_id, mask):
        # ----------------------------------------------------------------
        # Pass inputs through BERT
        # We don't fine tune anything in BERT so we just pass the input
        # and take the pooled output (2nd item in tuple) and then pass it
        # through the fully connected layers
        outputs = self.electra(sent_id, attention_mask=mask)
        cls_hs = outputs[0][:, 0, :]
        # ----------------------------------------------------------------
        # Pass through fully connected layers
        # Hidden layer
        x = self.fc(cls_hs)    # Fully connected layer
        x = self.actvfunc(x)        # Activation function
        # Dropout
        x = self.dropout(x)
        # Output layer
        # MULTILABEL CLASSIFICATION
        # x = self.outp(x)
        # x = self.logsoftmax(x)    # Softmax function for multi-class classification
        # BINARY CLASSIFICATION
        x = self.outp(x)
        x = self.sigmoid(x)
        x = x.squeeze(-1)
        # ----------------------------------------------------------------

        return x

def train(model, train_dataloader):
    # Set model to training mode
    model.train()

    total_loss = 0
    total_preds = []
    total_labels = []

    for step, batch in enumerate(train_dataloader):
        # Move batch to GPU (if it is the device we have)
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch

        # Clear previously calculated gradients
        model.zero_grad()

        # Get model predictions and compute loss
        preds = model(sent_id, mask)
        loss = loss_func(preds, labels)

        total_loss += loss.item()

        # Backward pass and gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update optimizer
        optimizer.step()

        # Collect predictions and labels
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        # Store predictions and labels to calculate accuracy after all batches are processed
        total_preds.append(preds)
        total_labels.append(labels)

    # Concatenate accumulated predictions and labels
    total_preds = np.concatenate(total_preds, axis=0)
    total_labels = np.concatenate(total_labels, axis=0)

    # Convert predictions to binary labels for accuracy calculation
    train_labels = (total_preds >= 0.5).astype(int)
    train_accuracy = np.mean(train_labels == total_labels)

    avg_loss = total_loss / len(train_dataloader)
    return avg_loss, train_accuracy, total_preds

def evaluate(model, val_dataloader):
    # Set model to evaluation mode
    model.eval()

    total_loss = 0
    total_preds = []
    for step, batch in enumerate(val_dataloader):
        # Move batch to GPU
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch

        # Deactivate autograd and compute predictions
        # We deactivate autograd because we don't want to update the model's weights
        with torch.no_grad():
            preds = model(sent_id, mask)
            loss = loss_func(preds, labels)
            total_loss += loss.item()

            # Calculate predictions and move to CPU to detach from computational graph
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)

    avg_loss = total_loss / len(val_dataloader)
    total_preds = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds

# =========================================================================================================
# ==================================== Searching Loop =====================================================
# =========================================================================================================

# 3 Cross-validation
kf = KFold(n_splits=5,
           shuffle=True,
           random_state=42)

# Initialize variables to store the best hyperparameters
best_acc = 0
best_hyperparams = None

# Tokenize and encode the full train set
tokens_train = tokenize_and_encode(train_text)
train_seq, train_mask, train_y = to_tensor(tokens_train, train_labels)

# Iterate over all combinations of hyperparameters and model configurations
for params in ParameterGrid(param_grid):
    fold_accuracies = []
    fold_f1s = []
    write_log("---------------------------------------------")
    write_log(f"Evaluating combination: {params}")
    write_log("---------------------------------------------")

    fold_number = 1  # Initialize fold counter
    # Perform 3CV on the 90% training set
    for train_index, val_index in kf.split(train_seq):
        write_log(f"\n Fold #{fold_number}")
        write_log("---------------------")

        # Get training and validation folds
        train_seq_fold = train_seq[train_index]
        train_mask_fold = train_mask[train_index]
        train_y_fold = train_y[train_index]

        val_seq_fold = train_seq[val_index]
        val_mask_fold = train_mask[val_index]
        val_y_fold = train_y[val_index]

        # Create data loaders for train and validation folds
        train_dataloader = create_data_loader(train_seq_fold, train_mask_fold, train_y_fold, RandomSampler(train_seq_fold), params['batch_size'])
        val_dataloader = create_data_loader(val_seq_fold, val_mask_fold, val_y_fold, SequentialSampler(val_seq_fold), 32)

        # Re-Initialize the model and move it to GPU
        # Load the untrained model to be retrained (from the begining) in the current fold.
        # bert = AutoModel.from_pretrained('bert-base-uncased')
        # deberta = AutoModel.from_pretrained('microsoft/deberta-base')
        # albert = AlbertModel.from_pretrained("albert-base-v2")
        # roberta = AutoModel.from_pretrained('roberta-base')
        electra = AutoModel.from_pretrained('google/electra-base-discriminator')

        model = ELECTRA_Arch(electra, params['dropout_p'], params['hidden_size'], params['activation_fn'])
        model = model.to(device)

        # Define optimizer and class weights
        optimizer = AdamW(model.parameters(),
                        lr=params['learning_rate'],
                        weight_decay=params['weight_decay'],
                        no_deprecation_warning=True)

        # Compute class weights and define loss function
        # We compute the class weights to handle the class imbalance problem in the dataset
        # so that we reduce the bias towards the majority class
        class_wts = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(train_y_fold.numpy()),
                                         y=train_y_fold.numpy())
        # weights = torch.tensor(class_wts, dtype=torch.float).to(device)
        # loss_func = nn.NLLLoss(weight=weights)
        # OR
        weights  = torch.tensor(class_wts[1], dtype=torch.float).to(device)  # Weight for positive class only
        loss_func = nn.BCEWithLogitsLoss(pos_weight=weights)

        best_valid_loss = float('inf')
        bestEpochModel = None

        # Training loop
        train_preds, val_preds = [], []
        for epoch in range(params['epochs']):
            write_log(f'\n Epoch {epoch + 1} / {params["epochs"]}')

            # Train the model on the training set
            train_loss, train_accuracy, train_labels = train(model, train_dataloader)

            # Evaluate the model on the validation set
            val_loss, val_preds = evaluate(model, val_dataloader)

            # Save the model and the metrics of the current model for the best epochs
            if val_loss <= best_valid_loss:   # If we find one with the same, keep the one with the biggest epoch
                bestEpochModel = model

                # Convert predictions to class labels for validation
                # MULTILABEL CLASSIFICATION
                # train_labels = np.argmax(train_preds, axis=1)
                # val_labels = np.argmax(val_preds, axis=1)
                # BINARY CLASSIFICATION
                val_labels = (val_preds >= 0.5).astype(int)

                # Compute accuracy for training and validation
                val_accuracy = np.mean(val_labels == val_y_fold.numpy())
                val_f1 = f1_score(val_labels, val_y_fold.numpy(), average='weighted')

            # Print losses and accuracies for this epoch
            # Log both training and validation accuracy per epoch to identify overfitting patterns.
            write_log(f'Training   -> Loss: {train_loss:.4f} | Accuracy: {train_accuracy:.4f}')
            write_log(f'Validation -> Loss: {val_loss:.4f} | Accuracy: {val_accuracy:.4f} | F1-Score: {val_f1:.4f}')

        fold_accuracies.append(val_accuracy)
        fold_f1s.append(val_f1)

        # Increment the fold counter
        fold_number += 1

        del electra
        del model
        del optimizer
        del class_wts
        del weights
        del loss_func
        del train_dataloader
        del val_dataloader
        torch.cuda.empty_cache()    # Free up GPU memory
        gc.collect()                # Forces garbage collection

    # Calculate average accuracy across folds for the current hyperparameter set
    avg_fold_accuracy = np.mean(fold_accuracies)
    avg_fold_f1s = np.mean(fold_f1s)
    write_log(f'Average Cross-Validation Accuracy: {avg_fold_accuracy:.4f}')
    write_log(f'Average Cross-Validation F1-Score: {avg_fold_f1s:.4f}')

    # If this combination of hyperparameters performs the best, save it
    if avg_fold_accuracy > best_acc:
        best_acc = avg_fold_accuracy
        best_f1 = avg_fold_f1s
        best_hyperparams = params

        # Saving the metrics and params into a file
        data = {'params': best_hyperparams,
                'accuracy': best_acc,
                'f1-score': best_f1
                }

        # Writing data to a text file in JSON format
        with open(base_path + model_name + '_BestHyperparameters_' + subjob_name + '.txt', 'w') as file:
            json.dump(data, file, indent=4)  # indent for pretty-printing

# =========================================================================================================
# =========================================================================================================
# =========================================================================================================

write_log("\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")
write_log(f"Best Accuracy: {best_acc:.4f}")
write_log(f"Best F1-Score: {best_f1:.4f}")
write_log(f"Best Hyperparameters: {best_hyperparams}")
write_log("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")