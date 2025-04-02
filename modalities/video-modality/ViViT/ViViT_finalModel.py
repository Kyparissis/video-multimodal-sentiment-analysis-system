import pathlib
from transformers import VivitImageProcessor, VivitForVideoClassification
import os
import torch
import torch.nn as nn
from collections import OrderedDict
import gc
import json
import time
from functools import partial
import itertools

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(device)

# FIX FOR IMPORT ISSUE FROM HERE: https://github.com/xinntao/Real-ESRGAN/issues/768
import sys
import types
from torchvision.transforms.functional import rgb_to_grayscale

# Create a module for `torchvision.transforms.functional_tensor`
functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
functional_tensor.rgb_to_grayscale = rgb_to_grayscale

# Add this module to sys.modules so other imports can access it
sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

dataset_root_path = "/home/k/kyparkypar/ondemand/data/sys/myjobs/projects/default/dataset/CMU-MOSI/Raw_reorganised/"
dataset_root_path = pathlib.Path(dataset_root_path)

all_video_file_paths = (
    list(dataset_root_path.glob("train/*/*.mp4"))
    + list(dataset_root_path.glob("valid/*/*.mp4"))
    + list(dataset_root_path.glob("test/*/*.mp4"))
)

class_labels = sorted({str(path).split("/")[-2] for path in all_video_file_paths})

label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}

print(f"Unique classes: {list(label2id.keys())}.")

model_ckpt = "google/vivit-b-16x2-kinetics400"
image_processor = VivitImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = VivitForVideoClassification.from_pretrained(
    model_ckpt,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)

model.init_weights()

batch_size = 2
gradient_accumulation_steps = 8

learning_rate = 5e-6
weight_decay = 1e-4

max_epochs = 2

hidden_size = 64
dropout_p = 0.1
activation_fn = "tanh"

if activation_fn == 'relu':
    activation_function = nn.ReLU()
elif activation_fn == 'tanh':
    activation_function = nn.Tanh()
elif activation_fn == 'No':
    activation_function = nn.Identity()

model.classifier = nn.Sequential(
    OrderedDict([
        ('dense', nn.Linear(768, hidden_size)),
        ('act_func', activation_function),
        ('dropout', nn.Dropout(dropout_p)),
        ('dense_outp', nn.Linear(hidden_size, model.config.num_labels)),
    ])
)

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

mean = image_processor.image_mean
std = image_processor.image_std
if "shortest_edge" in image_processor.size:
    height = width = image_processor.size["shortest_edge"]
else:
    height = image_processor.size["height"]
    width = image_processor.size["width"]
resize_to = (height, width)

num_frames_to_sample = model.config.num_frames

print("Num of frames to sample: ", num_frames_to_sample)


sample_rate = 4
fps = 30
clip_duration = num_frames_to_sample * sample_rate / fps

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

train_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, "train"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=train_transform,
)

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

val_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, "valid"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=val_transform,
)

test_dataset = pytorchvideo.data.Ucf101(
    data_path=os.path.join(dataset_root_path, "test"),
    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
    decode_audio=False,
    transform=val_transform,
)

import imageio
import numpy as np
from IPython.display import Image

# from pytorchvideo.data.clip_sampling import UniformClipSampler

# # Set stride to a large value (greater than any video length)
# clip_sampler = UniformClipSampler(clip_duration=clip_duration, stride=clip_duration)

# test_dataset = pytorchvideo.data.Ucf101(
#     data_path=os.path.join(dataset_root_path, "test"),
#     clip_sampler=clip_sampler,
#     decode_audio=False,
#     transform=val_transform,
# )

from transformers import TrainingArguments, Trainer, TrainerCallback

model_name = model_ckpt.split("/")[-1]
new_model_name = f"{model_name}-finetuned-cmu-mosi"

# Configure training run with TrainingArguments class      
metric_for_best_model = "loss"   # Save the model and the metrics of the current model for the best epochs
training_args = TrainingArguments(
    output_dir="./runs/vivit_classifier",
    # logging_dir="./logs/vivit_classifier",
    # report_to="tensorboard",
    learning_rate=learning_rate,                   
    push_to_hub=False,
    num_train_epochs=max_epochs,                   
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=batch_size,
    eval_strategy="epoch",                       
    save_strategy="no",
    # save_total_limit=0,  # Ensure no checkpoints are saved
    # eval_steps=1,
    # save_steps=1,
    weight_decay=weight_decay,
    load_best_model_at_end=False,
    # metric_for_best_model=metric_for_best_model,
    remove_unused_columns=False,
    # eval_accumulation_steps=eval_accumulation_steps,
    # logging_strategy="epoch",
    logging_steps=10,
    lr_scheduler_type="constant",  # Ensures no decay in learning rate
    fp16=True,
    max_steps=(train_dataset.num_videos // (batch_size * gradient_accumulation_steps)) * max_epochs,
)

class_wts = np.array([1.16304348, 0.87704918])

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
        self.curr_epoch = 1
        self.best_epoch = None
        self.training_metrics = []  # Track training loss at the end of each epoch
        self.eval_metrics = []      # Track evaluation loss at the end of each epoch
                        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and metric_for_best_model == "loss":
            if "eval_loss" in metrics:
                # print(state.epoch)
                # print(round(state.epoch))
                self.eval_metrics.append((self.curr_epoch, metrics["eval_loss"]))
                current_loss = metrics["eval_loss"]
                # print(f"Epoch #{int(state.epoch)} | Validation Loss: {current_loss:.5f} | Validation Accuracy: {metrics['eval_accuracy']:.5f}")
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.best_epoch = self.curr_epoch
                    self.best_acc = metrics["eval_accuracy"]
                    
        elif metrics is not None and metric_for_best_model == "accuracy":
            if "eval_loss" in metrics:
                self.eval_metrics.append((self.curr_epoch, metrics["eval_loss"]))
                current_acc = metrics["eval_accuracy"]
                # print(f"Epoch #{int(state.epoch)} | Validation Accuracy: {metrics['eval_accuracy']:.5f} | Validation Loss: {current_loss:.5f}")
                if current_acc > self.best_acc:
                    self.best_acc = current_acc
                    self.best_epoch = self.curr_epoch
                    self.best_loss = metrics["eval_loss"]
                    
        self.curr_epoch += 1

    def on_epoch_end(self, args, state, control, **kwargs):
        # Log training loss at the end of the epoch
        if state.log_history:
            # Extract the last logged loss
            for log in reversed(state.log_history):
                if "loss" in log:
                    self.training_metrics.append((state.epoch, log["loss"]))
                    break

    # def on_log(self, args, state, control, logs=None, **kwargs):
    #     if logs and "loss" in logs:
    #         self.training_metrics.append((state.epoch, logs["loss"]))

best_model_callback = BestModelEpochCallback()

import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

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

trainer = CustomTrainer(
    model,
    training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=image_processor,
    callbacks=[best_model_callback],  # Not used during CV, only here to find optimal epochs
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

trainer.train()

print("Optimal Epochs: ", (round(best_model_callback.best_epoch)))

def run_inference(model, video_or_dataset):
    """
    Run inference on either a single video or a dataset of videos.
    
    Args:
        model (torch.nn.Module): The model to use for inference.
        video_or_dataset (Union[torch.Tensor, LabeledVideoDataset]): 
            A single video tensor or a dataset of videos.
    
    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, List[int]]]: 
            If a single video tensor is provided, returns logits (torch.Tensor).
            If a dataset is provided, returns a tuple containing:
            - logits for all videos (torch.Tensor)
            - a list of corresponding labels (List[int]).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Case 1: Single video input
    if isinstance(video_or_dataset, torch.Tensor):
        # (num_frames, num_channels, height, width) -> (num_channels, num_frames, height, width)
        permuted_video = video_or_dataset.permute(1, 0, 2, 3)
        inputs = {"pixel_values": permuted_video.unsqueeze(0).to(device)}
        
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.logits

    # Case 2: Dataset input
    logits_list = []
    labels_list = []
    dataset_iterator = iter(video_or_dataset)  # Create an iterator for the dataset
    i = 0
    for i in range(video_or_dataset.num_videos):
        sample = next(dataset_iterator)  # Get the next sample
        video = sample["video"]
        label = sample["label"]  # Extract label
        permuted_video = video.permute(1, 0, 2, 3)
        inputs = {"pixel_values": permuted_video.unsqueeze(0).to(device)}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits_list.append(outputs.logits)
        labels_list.append(label)  # Append label to labels_list
        print(i, "->", outputs.logits, "-", label)
        i += 1
    
    return torch.cat(logits_list, dim=0), labels_list

logits = run_inference(model, test_dataset)

real_labels = logits[1]

predicted_logits = logits[0]
predicted_labels = predicted_logits.argmax(-1).cpu().numpy()

from sklearn.metrics import classification_report

print(classification_report(real_labels, predicted_labels, digits=4))

# Convert lists to tensors
predicted_logits_tensor = torch.tensor(predicted_logits, dtype=torch.float).to(device)
real_labels_tensor = torch.tensor(real_labels, dtype=torch.long).to(device)

# Compute class weights
weights = torch.tensor(class_wts, dtype=torch.float).to(device)
loss_fct = nn.CrossEntropyLoss(weight=weights)

# Compute loss
loss = loss_fct(predicted_logits_tensor.view(-1, predicted_logits_tensor.size(-1)), real_labels_tensor.view(-1))

print("Avg. test loss: ", loss.item())

import pandas as pd

confusion_matrix = pd.crosstab(real_labels, predicted_labels)

import seaborn as sns
import matplotlib.pyplot as plt

# Set the size of the figure
plt.figure(figsize=(10, 7))

# Create a heatmap from the confusion matrix
sns.heatmap(confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=True)

# Set titles and labels
plt.title('Fine-Tuned ViViT (Optimal Parameters) Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Show the plot
# plt.show()
plt.savefig('ViViTConfusionMatrix.png')

