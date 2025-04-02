# -*- coding: utf-8 -*-
"""Data2Vec_searchParams (1).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Kn_xoSlEmfJVclJViBRXRvH_e94RXnZl

# Data2Vec

## Libraries
"""

from datasets import Dataset, Audio, ClassLabel, Features, Value
import pandas as pd
import torch
import torch.nn as nn
import evaluate
import numpy as np
import warnings
from transformers import Trainer, TrainingArguments, TrainerCallback
from audiomentations import Compose, AddGaussianSNR, GainTransition, Gain, ClippingDistortion, TimeStretch, PitchShift
from transformers import AutoFeatureExtractor, Data2VecAudioConfig, Data2VecAudioForSequenceClassification
from copy import deepcopy
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, ParameterGrid
from collections import OrderedDict
import gc
import json
import time

subjob_name = 'full'
model_name = "Data2Vec"
base_path = "/home/k/kyparkypar/ondemand/data/sys/myjobs/projects/default/sound-modality/" + model_name + "/"
dataset_base_path = "/home/k/kyparkypar/ondemand/data/sys/myjobs/projects/default/dataset/CMU-MOSI/"

# Log file path
log_file_path = base_path + model_name + '_TrainingLog_' + subjob_name + '.txt'

# Function to write logs
def write_log(message):
    print(message)
    with open(log_file_path, 'a') as log_file:  # Open file in append mode
        log_file.write(message + '\n')  # Write the message and add a newline

"""## Ensure GPU access"""

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

write_log(f'Using device: {device}')

"""## Load dataset"""

# Define class labels
class_labels = ClassLabel(names=["Negative Sentiment", "Positive Sentiment"])

# Define features with audio and label columns
features = Features({
    "audio": Audio(),        # Define the audio feature
    "labels": class_labels,  # Assign the class labels
})

label2id = {
    "Negative Sentiment": 0,
    "Positive Sentiment": 1
}

# Load and preprocess the CSV file
df = pd.read_csv(dataset_base_path + "label_edited.csv",
                 sep='\t',
                 encoding='utf-8',
                 header=0)

df = df[['video_id', 'clip_id', 'annotation_label', 'mode']].dropna()

# This function is used to return the full path of a video in the CMU-MOSI dataset format
def get_audio_path(video_id, clip_id):
    return dataset_base_path + f"Splited/Raw_onlyAudio/{video_id}/{clip_id}.wav"

"""#### Get audio paths and labels of the full dataset"""

# Construct audio file paths and labels
audio_paths = df.apply(lambda row: get_audio_path(row['video_id'], row['clip_id']), axis=1).tolist()
labels = df['annotation_label'].astype(int).tolist()  # Convert labels to integers if necessary

"""Assure they are loaded correctly"""

# dataset = Dataset.from_dict({
#     "audio": audio_paths,
#     "labels": labels,
# }, features=features)

# dataset

# # Verify the dataset structure
# print(dataset[0])

"""## Load feature extractor"""

# we define which pretrained model we want to use and instantiate a feature extractor
pretrained_model = "facebook/data2vec-audio-base-960h" 
feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model,
                                                         return_attention_mask=True
                                                         )

"""### View extractor's inputs"""

# we save model input name and sampling rate for later use
model_input_name = 'input_values'  # key -> 'input_values'
SAMPLING_RATE = feature_extractor.sampling_rate

"""### Set maximum audio length to be processed

- Below this value, the audio will be padded.
- After this value, the audio will be truncated
"""

MAX_AUDIO_SECONDS = 10  # Maximum audio duration in seconds

MAX_AUDIO_LENGTH = SAMPLING_RATE * MAX_AUDIO_SECONDS

"""## Dataset Preprocess"""

def preprocess_audio(batch):
    wavs = [audio["array"] for audio in batch["input_values"]]
    
    # inputs are spectrograms as torch.tensors now
    inputs = feature_extractor(wavs, 
                               sampling_rate=SAMPLING_RATE,
                               max_length=MAX_AUDIO_LENGTH,
                               padding="max_length",
                               truncation=True,
                               return_tensors="pt")
    # print(inputs)
    output_batch = {"input_values": inputs.input_values, 
                    "attention_mask": inputs.attention_mask,
                    "labels": list(batch["labels"])
                   }
    
    return output_batch

"""### Augmentations"""

# audio_augmentations = Compose([
##     Adds Gaussian noise to the audio while maintaining a specified signal-to-noise ratio (SNR)
#     AddGaussianSNR(min_snr_db=10,    # min_snr_db: Minimum SNR in decibels (dB). A lower value means more noise.
#                    max_snr_db=20),   # max_snr_db: Maximum SNR in dB. A higher value means less noise.
##     Adjusts the audio volume by applying a uniform gain
#     Gain(min_gain_db=-6,      # min_gain_db: Minimum gain in dB. A negative value decreases volume.
#          max_gain_db=6),      # max_gain_db: Maximum gain in dB. A positive value increases volume.
##     Gradually applies gain changes over a specified duration to create a smooth volume transition.
#     GainTransition(min_gain_db=-6,                     # min_gain_db: Minimum gain change in dB. Negative for fading out, positive for fading in.
#                    max_gain_db=6,                      # max_gain_db: Maximum gain change in dB.
#                    min_duration=0.01,                  # min_duration: Minimum duration of the gain transition (in seconds or as a fraction of total duration, depending on duration_unit
#                    max_duration=0.3,                   # max_duration: Maximum duration of the gain transition.
#                    duration_unit="fraction"),          # "fraction": Durations are a fraction of the total audio length.
#                                                        # "seconds": Durations are in absolute time.
##     Simulates distortion by artificially clipping the waveform
#     ClippingDistortion(min_percentile_threshold=0,     # min_percentile_threshold: Minimum amplitude threshold percentile for clipping.
#                        max_percentile_threshold=30,    # max_percentile_threshold: Maximum amplitude threshold percentile for clipping. A higher value clips more of the waveform.
#                        p=0.5),                         p: Probability of applying this augmentation.
##     Alters the playback speed of the audio without changing its pitch
#     TimeStretch(min_rate=0.8,       # min_rate: Minimum playback speed (as a fraction of the original). Values <1 slow down the audio.
#                 max_rate=1.2),      # max_rate: Maximum playback speed. Values >1 speed up the audio.
##     Changes the pitch of the audio without altering its speed
#     PitchShift(min_semitones=-4,    # min_semitones: Minimum pitch shift in semitones (negative values lower the pitch).
#                max_semitones=4),    # max_semitones: Maximum pitch shift in semitones (positive values raise the pitch).
# ], p=0.8, shuffle=True)  # p: Overall probability of applying the composed augmentations
#                          # shuffle: Randomly shuffles the order of augmentations during each application.

# def preprocess_audio_with_transforms(batch):
#     # we apply augmentations on each waveform
#     wavs = [audio_augmentations(audio["array"], sample_rate=SAMPLING_RATE) for audio in batch["input_values"]]
#     inputs = feature_extractor(wavs, sampling_rate=SAMPLING_RATE, return_tensors="pt")

#     output_batch = {model_input_name: inputs.get(model_input_name), "labels": list(batch["labels"])}

#     return output_batch

# # Cast the audio column to the appropriate feature type and rename it
# dataset = dataset.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))
# dataset = dataset.rename_column("audio", "input_values")

# # Apply the transformation to the whole dataset (Train, Validation, Test)
# dataset = dataset.rename_column("audio", "input_values")  # rename audio column
# dataset.set_transform(preprocess_audio, output_all_columns=True)

"""### Dataset's split

Here we use the dataset's preset split, in order to compare we other models
"""

train_df = df[df['mode'] == 'train']
# valid_df = df[df['mode'] == 'valid']
# test_df = df[df['mode'] == 'test']

def create_dataset_from_df(df, indexes=None):
    audio_paths = df.apply(lambda row: get_audio_path(row['video_id'], row['clip_id']), axis=1).tolist()
    labels = df['annotation_label'].astype(int).tolist()

    if indexes != None:
        audio_paths = [audio_paths[i] for i in indexes]
        labels = [labels[i] for i in indexes]
    
    return Dataset.from_dict({
        "audio": audio_paths,
        "labels": labels,
    }, features=features)


feature_extractor.mean = -7.9245896
feature_extractor.std = 5.2356324

feature_extractor.do_normalize = True # we set normalization to true back again

torch.cuda.empty_cache()    # Free up GPU memory
gc.collect()                # Forces garbage collection

"""## Hyperparameter Search - Cross-Validation

### Metrics calculation
"""

accuracy = evaluate.load("accuracy")
# recall = evaluate.load("recall")
# precision = evaluate.load("precision")
# f1 = evaluate.load("f1")
AVERAGE = "binary"    # Default behavior for binary classification

def compute_metrics(eval_pred):   
    logits = eval_pred.predictions
    
    if isinstance(logits, tuple):
        logits = logits[0]

    # Handle different shapes of logits
    if logits.ndim == 1:  # Binary classification or single-class scores
        predictions = (logits > 0).astype(int)  # Convert to binary predictions
    else:  # Multiclass classification
        predictions = np.argmax(logits, axis=1)
    
    metrics = {}
    metrics.update(accuracy.compute(predictions=predictions, references=eval_pred.label_ids))
    # metrics.update(precision.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
    # metrics.update(recall.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
    # metrics.update(f1.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
    # metrics['weighted_f1'] = f1.compute(predictions=predictions, references=eval_pred.label_ids, average="weighted")['f1']
    return metrics

# eval_accumulation_steps = 128

param_grid = {
    'epochs': [5],
    'hidden_size': [32, 64, 128],
    'gradient_accumulation_steps': [8],
    'batch_size': [2, 4],
    'dropout_p': [0.3],
    'learning_rate': [1e-5, 5e-6, 1e-6],
    'activation_fn': ['tanh', 'relu'],
    'weight_decay': [1e-4, 1e-5]
}

# 5 Fold Cross-validation
k = 5
kf = KFold(n_splits=k,
           shuffle=True,
           random_state=42)

# Initialize variables to store the best hyperparameters
best_acc = 0
best_hyperparams = None

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        
        outputs = model(
            input_values=inputs["input_values"],
            attention_mask=inputs.get("attention_mask"),
            labels=labels
        )

        logits = outputs.get("logits")

        # Compute custom loss with class weights
        weights = torch.tensor(class_wts, dtype=torch.float).to(device)
        loss_fct = nn.CrossEntropyLoss(weight=weights)

        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

class BestModelEpochCallback(TrainerCallback):
    def __init__(self):
        self.best_loss = float("inf")
        self.best_acc = 0.0
        self.best_epoch = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and metric_for_best_model == "loss":
            if "eval_loss" in metrics:
                current_loss = metrics["eval_loss"]
                write_log(f"Epoch #{int(round(state.epoch))} | Validation Loss: {current_loss:.5f} | Validation Accuracy: {metrics['eval_accuracy']:.5f}")
                if current_loss < self.best_loss and round(state.epoch) > 1:
                    self.best_loss = current_loss
                    self.best_epoch = round(state.epoch)
                    self.best_acc = metrics["eval_accuracy"]
        elif metrics is not None and metric_for_best_model == "accuracy":
            if "eval_loss" in metrics:
                current_acc = metrics["eval_accuracy"]
                write_log(f"Epoch #{int(round(state.epoch))} | Validation Accuracy: {metrics['eval_accuracy']:.5f} | Validation Loss: {current_loss:.5f}")
                if current_acc > self.best_acc and round(state.epoch) > 1:
                    self.best_acc = current_acc
                    self.best_epoch = round(state.epoch)
                    self.best_loss = metrics["eval_loss"]

warnings.filterwarnings("ignore", category=UserWarning, message=".*Converting to np.float32.*")
warnings.filterwarnings("ignore", message=".*is ill-defined.*")
warnings.filterwarnings("ignore", message=".*were not initialized from the model checkpoint.*")

# Load configuration from the pretrained model
config = Data2VecAudioConfig.from_pretrained(pretrained_model)
# Update configuration with the number of labels in our dataset
config.num_labels = 2
config.label2id = label2id
config.id2label = {v: k for k, v in label2id.items()}

train_labels = np.asarray(train_df['annotation_label'].astype(int).tolist())

for params in ParameterGrid(param_grid):
    fold_accuracies = []
    # fold_f1s = []

    write_log("---------------------------------------------")
    write_log(f"Evaluating combination: {params}")
    write_log("---------------------------------------------")

    if params['activation_fn'] == 'relu':
        activation_function = nn.ReLU()
    elif params['activation_fn'] == 'tanh':
        activation_function = nn.Tanh()
    elif params['activation_fn'] == 'No':
        activation_function = nn.Identity()

    # copy_fullTrainDataset = copy.deepcopy(dataset["train"])
    # print(id(dataset["train"]))
    fold_number = 1  # Initialize fold counter
    times = []
    for train_index, val_index in kf.split(train_df['annotation_label'].astype(int).tolist()):

        write_log(f"\n Fold #{fold_number} / {k}")
        write_log("---------------------")
        
        start = time.time()

        train_dataset = create_dataset_from_df(train_df, train_index.tolist()).rename_column("audio", "input_values")  # rename audio column
        valid_dataset = create_dataset_from_df(train_df, val_index.tolist()).rename_column("audio", "input_values") 

        dataset = {
            'train': train_dataset,
            'validation': valid_dataset,
            # 'test': test_dataset
        }
        
        dataset["train"] = dataset["train"].cast_column("input_values", Audio(sampling_rate=feature_extractor.sampling_rate))
        dataset["validation"] = dataset["validation"].cast_column("input_values", Audio(sampling_rate=feature_extractor.sampling_rate))
        
        # current_trainSet = dataset["train"].select(train_index)
        # # print(id(current_trainSet))
        # current_valSet = dataset["train"].select(val_index)
                
        dataset["train"].set_transform(preprocess_audio, 
                               output_all_columns=False)

        dataset["validation"].set_transform(preprocess_audio, 
                               output_all_columns=False)

        # current_train_labels = np.asarray(train_labels)
        class_wts = compute_class_weight(class_weight='balanced',
                                 classes=np.unique(train_labels),
                                 y=train_labels)
        # del current_train_labels
        
        # Initialize the model with the updated configuration
        model = Data2VecAudioForSequenceClassification.from_pretrained(pretrained_model, 
                                                                        config=config, 
                                                                        ignore_mismatched_sizes=True)

        model.init_weights()

        model.classifier = nn.Sequential(
            OrderedDict([
                ('dense', nn.Linear(256, params['hidden_size'])),
                ('act_func', activation_function),
                ('dropout', nn.Dropout(params['dropout_p'])),
                ('dense_outp', nn.Linear(params['hidden_size'], model.config.num_labels)),
            ])
        )
        
        # Configure training run with TrainingArguments class      
        metric_for_best_model = "loss"   # Save the model and the metrics of the current model for the best epochs
        
        training_args = TrainingArguments(
            output_dir="./runs/data2vec_classifier",
            overwrite_output_dir=True,
            # logging_dir="./logs/data2vec_classifier",
            report_to="tensorboard",
            disable_tqdm=True,
            learning_rate=params['learning_rate'],                   
            push_to_hub=False,
            num_train_epochs=params['epochs'],                   
            per_device_train_batch_size=params['batch_size'],
            gradient_accumulation_steps=params['gradient_accumulation_steps'],
            per_device_eval_batch_size=1,
            eval_strategy="epoch",                       
            save_strategy="no",
            save_total_limit=0,  # Ensure no checkpoints are saved
            eval_steps=1,
            save_steps=1,
            weight_decay=params['weight_decay'],
            load_best_model_at_end=False,
            metric_for_best_model=metric_for_best_model,
            remove_unused_columns=False,
            # eval_accumulation_steps=eval_accumulation_steps,
            logging_strategy="epoch",
            # logging_steps=20,
            lr_scheduler_type="constant",  # Ensures no decay in learning rate
            # fp16=True,
        )
        
        best_model_callback = BestModelEpochCallback()

        # Setup the trainer
        trainer_new = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            callbacks=[best_model_callback],  # Not used during CV, only here to find optimal epochs
            compute_metrics=compute_metrics,  # Use the metrics function from above
        )

        trainer_new.train()

        write_log(f"Optimal Epochs, for fold #{fold_number}: {int(best_model_callback.best_epoch)}")
        write_log(f"Best fold #{fold_number} accuracy: {best_model_callback.best_acc:.8f}")
        
        end = time.time()
        write_log(f"Time to train for {params['epochs']} epochs (1 fold - Fold #{fold_number}) = {(end - start)/60} min.")
        times.append(end - start)

        fold_number += 1
        
        fold_accuracies.append(best_model_callback.best_acc)
        
        del trainer_new
        del best_model_callback
        del training_args
        del model
        del class_wts
        del train_dataset
        del valid_dataset
        del dataset
        gc.collect()                # Forces garbage collection
        torch.cuda.empty_cache()    # Free up GPU memory
        gc.collect() 
    
    # Calculate average accuracy across folds for the current hyperparameter set
    avg_fold_accuracy = np.mean(fold_accuracies)
    write_log("\n===============================================")
    write_log(f"---> Combination's Average Cross-Validation Accuracy: {avg_fold_accuracy:.4f}")
    write_log(f"(Combination total time = {sum(times)/60} min)")
    write_log("===============================================\n\n")

    # If this combination of hyperparameters performs the best, save it
    if avg_fold_accuracy > best_acc:
        best_acc = avg_fold_accuracy
        best_hyperparams = params

        # Saving the metrics and params into a file
        data = {'params': best_hyperparams,
                'accuracy': best_acc,
                }
        # Writing data to a text file in JSON format
        with open(base_path + model_name + '_BestHyperparameters_' + subjob_name + '.txt', 'w') as file:
            json.dump(data, file, indent=4)  # indent for pretty-printing
    
    gc.collect()                # Forces garbage collection
    torch.cuda.empty_cache()    # Free up GPU memory
    gc.collect()                # Forces garbage collection


write_log("\n=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")
write_log(f"Best Accuracy: {best_acc:.4f}")
write_log(f"Best Hyperparameters: {best_hyperparams}")
write_log("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")
