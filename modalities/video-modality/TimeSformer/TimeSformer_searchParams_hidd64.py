# -*- coding: utf-8 -*-

import pandas as pd
import torch
# from torch.optim.lr_scheduler import LambdaLR
from transformers import Trainer, TrainingArguments
from torch.optim import AdamW
import torch.nn as nn
import evaluate
import numpy as np
import warnings
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers import AutoImageProcessor, TimesformerForVideoClassification
# from copy import deepcopy
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import KFold, ParameterGrid
from collections import OrderedDict
import gc
import os
import json
import time
from functools import partial
import itertools
import pathlib
# ===========================
# FIX FOR IMPORT ISSUE FROM HERE: https://github.com/xinntao/Real-ESRGAN/issues/768
import sys
import types
from torchvision.transforms.functional import rgb_to_grayscale

# Create a module for `torchvision.transforms.functional_tensor`
functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
functional_tensor.rgb_to_grayscale = rgb_to_grayscale

# Add this module to sys.modules so other imports can access it
sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor
# ===========================
import pytorchvideo.data
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

subjob_name = 'hidd64'
model_name = "TimeSformer"
base_path = "/home/k/kyparkypar/ondemand/data/sys/myjobs/projects/default/video-modality/" + model_name + "/"
# dataset_base_path = "/home/k/kyparkypar/ondemand/data/sys/myjobs/projects/default/dataset/CMU-MOSI/"
dataset_root_path = "/home/k/kyparkypar/ondemand/data/sys/myjobs/projects/default/dataset/CMU-MOSI/Raw_reorganised/"
dataset_root_path = pathlib.Path(dataset_root_path)
all_video_file_paths = (
    list(dataset_root_path.glob("train/*/*.mp4"))
    # + list(dataset_root_path.glob("valid/*/*.mp4"))
    # + list(dataset_root_path.glob("test/*/*.mp4"))
 )

class_labels = sorted({str(path).split("/")[-2] for path in all_video_file_paths})

label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}

class_wts = np.array([1.16304348, 0.87704918])

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

def collate_fn(examples):
    resize = Resize(resize_to)  # Ensure consistent size
    videos = []
    labels = []

    for example in examples:
        resized_video = resize(example["video"])  # Resize each video
        videos.append(resized_video.permute(1, 0, 2, 3))  # Permute dimensions
        labels.append(example["label"])

    pixel_values = torch.stack(videos)
    labels = torch.tensor(labels)

    return {"pixel_values": pixel_values, 
            "labels": labels}

model_ckpt = "facebook/timesformer-base-finetuned-k400"
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

mean = image_processor.image_mean
std = image_processor.image_std
if "shortest_edge" in image_processor.size:
    height = width = image_processor.size["shortest_edge"]
else:
    height = image_processor.size["height"]
    width = image_processor.size["width"]
resize_to = (height, width)

model = TimesformerForVideoClassification.from_pretrained(
    model_ckpt,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)
sample_rate = 4
fps = 30
num_frames_to_sample = model.config.num_frames
clip_duration = num_frames_to_sample * sample_rate / fps

del model

param_grid = {
    'epochs': [5],
    'hidden_size': [64],
    'gradient_accumulation_steps': [8],
    'batch_size': [2, 4],
    'dropout_p': [0.1],
    'learning_rate': [5e-5, 1e-5, 5e-6],
    # 'lr_scheduler_type': ['constant'],
    'activation_fn': ['tanh', 'relu'],
    'weight_decay': [1e-4, 1e-5]
}

train_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize(resize_to),
                ]
            ),
        ),
    ]
)

# train_transform_augm = Compose(
#     [
#         ApplyTransformToKey(
#             key="video",
#             transform=Compose(
#                 [
#                     UniformTemporalSubsample(num_frames_to_sample),
#                     Lambda(lambda x: x / 255.0),
#                     Normalize(mean, std),
#                     RandomShortSideScale(min_size=256, max_size=320),
#                     RandomCrop(resize_to),
#                     RandomHorizontalFlip(p=0.5),
#                 ]
#             ),
#         ),
#     ]
# )

val_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize(resize_to),
                ]
            ),
        ),
    ]
)

# def lr_lambda(current_step, steps_per_epoch, lr_scheduler_type):
#     epoch = current_step // steps_per_epoch

#     if lr_scheduler_type == "constant":
#         return 1.0
#     else:
#         if epoch <= 4:
#             return 1.0  # Keep learning rate constant for the first 5 epochs
#         # Halve every 2 epochs after epoch 5
#         elif epoch >= 5 and epoch <= 6:
#             return 0.5
#         elif epoch >= 7 and epoch <= 8:
#             return 0.5 / 2
#         elif epoch >= 9 and epoch <= 10:
#             return 0.5 / 4

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
        
        outputs = model(**inputs)

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
        self.curr_epoch = 1

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and metric_for_best_model == "loss":
            if "eval_loss" in metrics:
                current_loss = metrics["eval_loss"]
                write_log(f"Epoch #{int(self.curr_epoch)} | Validation Loss: {current_loss:.5f} | Validation Accuracy: {metrics['eval_accuracy']:.5f}")
                if current_loss < self.best_loss and round(state.epoch) > 1:
                    self.best_loss = current_loss
                    self.best_epoch = self.curr_epoch
                    self.best_acc = metrics["eval_accuracy"]
        elif metrics is not None and metric_for_best_model == "accuracy":
            if "eval_loss" in metrics:
                current_acc = metrics["eval_accuracy"]
                write_log(f"Epoch #{int(self.curr_epoch)} | Validation Accuracy: {metrics['eval_accuracy']:.5f} | Validation Loss: {current_loss:.5f}")
                if current_acc > self.best_acc and round(state.epoch) > 1:
                    self.best_acc = current_acc
                    self.best_epoch = self.curr_epoch
                    self.best_loss = metrics["eval_loss"]
        self.curr_epoch += 1
    
    # def on_epoch_end(self, args, state, control, **kwargs):
    #     optimizer = kwargs.get("optimizer")
    #     print(f"---> Used Learning rate: {optimizer.param_groups[0]['lr']}\n")

warnings.filterwarnings("ignore", category=UserWarning, message=".*Converting to np.float32.*")
warnings.filterwarnings("ignore", message=".*is ill-defined.*")
warnings.filterwarnings("ignore", message=".*were not initialized from the model checkpoint.*")

train_dataset = pytorchvideo.data.Ucf101(
    # TODO:
    data_path=os.path.join(dataset_root_path, "train"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=train_transform,
)

train_iterator = iter(train_dataset)
i = 0
fold_train_samples = []
fold_valid_samples = []

# for train_index, val_index in kf.split(all_video_file_paths):
#     # Extract train samples for the i+1-th fold
#     current_train_samples = [next(itertools.islice(train_iterator, idx, idx + 1)) for idx in train_index]
#     # Extract validation samples for the i+1-th fold
#     current_valid_samples = [next(itertools.islice(train_iterator, idx, idx + 1)) for idx in val_index]
    
#     # Save these samples to the respective i-th position
#     fold_train_samples.append(current_train_samples)
#     fold_valid_samples.append(current_valid_samples)
    
#     i += 1

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
        
    fold_number = 1  # Initialize fold counter
    times = []
    for train_index, val_index in kf.split(all_video_file_paths):

        write_log(f"\n Fold #{fold_number} / {k}")
        write_log("---------------------")
        
        start = time.time()
        
        # Define train and validation split file paths
        train_split_file = os.path.join(base_path, "splits", f"train_fold_{fold_number}.txt")
        val_split_file = os.path.join(base_path, "splits", f"test_fold_{fold_number}.txt")
        
        # Load training dataset for this fold
        train_dataset = pytorchvideo.data.Ucf101(
            data_path=train_split_file,
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
            decode_audio=False,
            transform=train_transform,
        )
        
        # Load validation dataset for this fold
        val_dataset = pytorchvideo.data.Ucf101(
            data_path=val_split_file,
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
            decode_audio=False,
            transform=val_transform,
        )

        # currFold_train_samples = fold_valid_samples[fold_train_samples-1]
        # currFold_valid_samples = fold_valid_samples[fold_number-1]
        
        # Initialize the model with the updated configuration
        model = TimesformerForVideoClassification.from_pretrained(
            model_ckpt,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )

        model.init_weights()

        model.classifier = nn.Sequential(
            OrderedDict([
                ('dense', nn.Linear(768, params['hidden_size'])),
                ('act_func', activation_function),
                ('dropout', nn.Dropout(params['dropout_p'])),
                ('dense_outp', nn.Linear(params['hidden_size'], model.config.num_labels)),
            ])
        )
        
        # optimizer = AdamW(model.parameters(), lr=params['learning_rate'])
        # steps_per_epoch = train_dataset_size // (params['batch_size'] * params['grad_accum_steps'])
        
        # lr_lambda_with_type = partial(lr_lambda, steps_per_epoch, lr_scheduler_type=params['lr_scheduler_type'])
        
        # scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_with_type)
        
        # Configure training run with TrainingArguments class      
        metric_for_best_model = "loss"   # Save the model and the metrics of the current model for the best epochs
        training_args = TrainingArguments(
            output_dir="./runs/timesformer_classifier-64",
            overwrite_output_dir=True,
            # logging_dir="./logs/videomae_classifier",
            report_to="tensorboard",
            disable_tqdm=True,
            learning_rate=params['learning_rate'],                   
            push_to_hub=False,
            num_train_epochs=params['epochs'],                   
            per_device_train_batch_size=params['batch_size'],
            gradient_accumulation_steps=params['gradient_accumulation_steps'],
            per_device_eval_batch_size=params['batch_size'],
            eval_strategy="epoch",                       
            save_strategy="no",
            save_total_limit=0,  # Ensure no checkpoints are saved
            # eval_steps=1,
            # save_steps=1,
            weight_decay=params['weight_decay'],
            load_best_model_at_end=False,
            metric_for_best_model=metric_for_best_model,
            remove_unused_columns=False,
            # eval_accumulation_steps=eval_accumulation_steps,
            # logging_strategy="epoch",
            logging_steps=10,
            lr_scheduler_type="constant",  # Ensures no decay in learning rate
            # fp16=True,
            max_steps=(len(train_index) // (params['batch_size'] * params['gradient_accumulation_steps'])) * params['epochs'],
        )
        
        best_model_callback = BestModelEpochCallback()

        # Setup the trainer
        trainer_new = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=image_processor,
            callbacks=[best_model_callback],  # Not used during CV, only here to find optimal epochs
            compute_metrics=compute_metrics,  # Use the metrics function from above
            data_collator=collate_fn,
        )

        trainer_new.train()

        write_log(f"Optimal Epochs, for fold #{fold_number}: {int(best_model_callback.best_epoch)}")
        write_log(f"Best fold #{fold_number} accuracy: {best_model_callback.best_acc:.8f}")
        
        end = time.time()
        write_log(f"Time to train for {params['epochs']} epochs (1 fold - Fold #{fold_number}) = {(end - start)/60} min.")
        times.append(end - start)

        fold_number += 1
        
        fold_accuracies.append(best_model_callback.best_acc)
        
        del train_split_file
        del val_split_file
        del train_dataset
        del val_dataset
        # del steps_per_epoch
        # del optimizer
        # del scheduler
        del trainer_new
        del best_model_callback
        del training_args
        del model
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
