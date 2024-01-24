import random
from torch.utils.data.sampler import Sampler
import torch
import torchvision  # torch package for vision related things
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For a nice progress bar!
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import math
from sklearn.preprocessing import StandardScaler
from torch import linalg as LA
import numpy as np
import math
import gensim
import os
import nltk
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
nltk.download('punkt')

# Read the stored model

save_path = './network.pth'
new_word2vec = gensim.models.Word2Vec.load("word2vec.model")
new_net = RNN(input_size, hidden_size, num_layers, num_classes, sequence_length)
new_net.load_state_dict(torch.load(save_path))
new_net.eval()

# Function to generate sessions for calculating the cluster centers for inference

def sessions_to_calcualte_cluster_centers_for_inference(seq_length,features):
  ignore = {".DS_Store", ".txt"}
  session_count=0
  for root, dirs, files in os.walk("sampled_sessions"):
    for filename in files:
        if filename not in ignore:
            with open(os.path.join(root,filename), 'r', encoding='utf8') as act_file:
              Lines = act_file.readlines()

              for line in Lines:

                session_count=session_count+1

  print('session_count: ', session_count)
  dataset=np.zeros((session_count, seq_length, features))
  session_label_old= []
  session_number=0
  for root, dirs, files in os.walk("sampled_sessions"):
    for filename in files:
        if filename not in ignore:
            with open(os.path.join(root,filename), 'r', encoding='utf8') as act_file:
              Lines = act_file.readlines()
              for line in Lines:
                acts, _, label = line.split(';')
                session_label_old.append(label.strip('\n'))
                sequence_number=0
                for act in acts.split(','):
                  if sequence_number<seq_length:
                    x=new_word2vec.wv.get_vector(act.lower())
                    for i in range(features):
                      dataset[session_number][sequence_number][i]=x[i]
                    sequence_number=sequence_number+1
                session_number=session_number+1

  session_label=np.array(session_label_old)
  session_label=session_label.astype(np.float32)
  dataset=dataset.astype(np.float32)
  return dataset, session_label

# generate the cluster centers for inference

data_vector, label= sessions_to_calcualte_cluster_centers_for_inference(seq_length, features)
dataset=Seq_Dataset(data_vector,label)
dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=len(dataset))
for batch_idx, (data, targets) in enumerate(tqdm(dataloader)):
  encode,_ = new_net(data)
V1,V0 = get_cluster_centers(encode, targets)

# function to create the test set

def test_dataset(seq_length,features):
  ignore = {".DS_Store", ".txt"}
  session_count=0
  for root, dirs, files in os.walk("test_data"):
    for filename in files:
        if filename not in ignore:
            with open(os.path.join(root,filename), 'r', encoding='utf8') as act_file:
              Lines = act_file.readlines()

              for line in Lines:

                session_count=session_count+1

  print('sesion_count: ', session_count)
  dataset=np.zeros((session_count, seq_length, features))
  session_label_old= []
  session_number=0
  for root, dirs, files in os.walk("test_data"):
    for filename in files:
        if filename not in ignore:
            with open(os.path.join(root,filename), 'r', encoding='utf8') as act_file:
              Lines = act_file.readlines()
              for line in Lines:
                acts, _, label = line.split(';')
                session_label_old.append(label.strip('\n'))
                sequence_number=0
                for act in acts.split(','):
                  if sequence_number<seq_length:
                    x=new_word2vec.wv.get_vector(act.lower())
                    for i in range(features):
                      dataset[session_number][sequence_number][i]=x[i]
                    sequence_number=sequence_number+1
                session_number=session_number+1

  session_label=np.array(session_label_old)
  session_label=session_label.astype(np.float32)
  dataset=dataset.astype(np.float32)
  return dataset, session_label

# generate the encoder representations of all the sessions in the test set

data_vector, label= test_dataset(seq_length, features)
dataset = Seq_Dataset(data_vector,label)
dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=len(dataset))
for batch_idx, (data, targets) in enumerate(tqdm(dataloader)):
  encode,_ = new_net(data)

# Function to calculate accuracy

def calculate_accuracy(data, target, V1, V0):

  count=0
  pred_label=np.zeros(target.shape[0], dtype=np.float32)
  for i in range(target.shape[0]):
    if LA.norm(data[i]-V1)<= LA.norm(data[i]-V0):
      pred_label[i]=1
    else:
      pred_label[i]=0


  pred_label=torch.from_numpy(pred_label)

  return pred_label, (torch.sum(torch.eq(pred_label, target))/target.shape[0])*100

# Function to calculate precision, recall, f1, and fpr scores

def calculate_performance_metrics(data, target, V1, V0):

  count=0
  pred_label=np.zeros(target.shape[0], dtype=np.float32)
  for i in range(target.shape[0]):
    if LA.norm(data[i]-V1)<= LA.norm(data[i]-V0):
      pred_label[i]=1
    else:
      pred_label[i]=0


  pred_label=torch.from_numpy(pred_label)

  tp = 0
  fp = 0
  fn = 0
  tn = 0

  for i in range(target.shape[0]):
    if (pred_label[i] == target[i]) & (target[i]==1):
      tp=tp+1
    elif (pred_label[i] != target[i]) & (target[i]==0):
      fp=fp+1
    elif (pred_label[i] != target[i]) & (target[i]==1):
      fn=fn+1
    elif (pred_label[i] == target[i]) & (target[i]==0):
      tn=tn+1


  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  f1 = 2*((precision*recall)/(precision+recall))
  fpr = fp/(tn+fp)


  return precision, recall, f1, fpr

# function to calculate auc scores

def calculate_auc_scores(data, target, pred_label, V1, V0):

  pred_label=np.zeros(target.shape[0], dtype=np.float32)
  for i in range(target.shape[0]):
    if LA.norm(data[i]-V1)<= LA.norm(data[i]-V0):
      pred_label[i]=1
    else:
      pred_label[i]=0

  target=target.detach().cpu().numpy()
  roc_auc = roc_auc_score(target, pred_label)
  avg_precision = average_precision_score(target, pred_label)

  return roc_auc, avg_precision

# obtain all the performance scores

pred_label, accuracy=calculate_accuracy(encode, targets, V1, V0)
precision, recall, f1, fpr = calculate_performance_metrics(encode, targets, V1, V0)
roc_auc, avg_precision = calculate_auc_scores(encode, targets, pred_label, V1, V0)
print(f"Accuracy: {accuracy}")
print(f"Precision, recall, f1, fpr: {precision, recall, f1, fpr}")
print(f"auc_roc, auc_pr: {roc_auc, avg_precision}")
