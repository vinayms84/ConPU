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

# Function for building word to vector representations of all the sessions

def build_word_to_vector():
  data = []
  Label= []
  ignore = {".DS_Store", ".txt"}
  session_count=0
  for root, dirs, files in os.walk("all_sessions"):
    for filename in files:
        if filename not in ignore:
            with open(os.path.join(root,filename), 'r', encoding='utf8') as act_file:
              Lines = act_file.readlines()

              for line in Lines:
                acts, _, label = line.split(';')
                temp = []
                session_count=session_count+1

                for act in acts.split(','):
                  temp.append(act.lower())

                data.append(temp)
                Label.append(label.strip('\n'))


  return data

seq_length=25
features=50

# construct the dataset to generate word to vector representations

data=build_word_to_vector()

# generate word to vector representations

word2Vec_model = gensim.models.Word2Vec(
                    data,
                    sg=1,
                    size=features,
                    min_count=1

                    )

# class for generating a sequence of activites in a session in tensor format

class Seq_Dataset(Dataset):
  def __init__(self, dataset, labels):

    labels=labels.flatten()
    self.x=torch.from_numpy(dataset[:,:,:])
    self.y=torch.from_numpy(labels).type(torch.LongTensor)
    self.n_samples=dataset.shape[0]


  def __getitem__(self,index):
    return self.x[index], self.y[index]

  def __len__(self):
    return self.n_samples

# class to define the LSTM network

input_size = features
hidden_size = features
num_layers = 2
num_classes = features
sequence_length = seq_length
learning_rate = 0.005
batch_size = 110
num_epochs = 10

# LSTM

class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length):
    super(RNN, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).float()



  def forward(self, x):
    # Set initial hidden and cell states
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
    # Forward propagate LSTM
    rnn_out, _ = self.lstm(x, (h0,c0))
    out = torch.mean(rnn_out, axis = 1)


    return out, _

# Initialization of the LSTM model and its optimizer

net = RNN(input_size, hidden_size, num_layers, num_classes, sequence_length)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# Function to generate cluster centers in each training batch

def get_cluster_centers(data, target, tau0=0.55,tau1=0.45):
  n=target.shape[0]
  count_1 = 0
  count_0 = 0
  data=data.detach().cpu().numpy()
  target=target.detach().cpu().numpy()
  for i in range(n):
    if target[i]==1:
      count_1 = count_1+1
    else:
      count_0 = count_0+1



  malicious = np.zeros([count_1,data.shape[1]], dtype=np.float32)
  train_set = np.zeros([count_0,data.shape[1]], dtype=np.float32)


  index_1=0
  index_0=0
  for k in range(n):
    if target[k]==1:
      malicious[index_1]=data[k]
      index_1=index_1+1
    else:
      train_set[index_0]=data[k]
      index_0 = index_0+1



  V1=np.mean(malicious, axis=0)
  VD=np.mean(train_set, axis=0)
  V0=(1/tau0)*(VD-(tau1*V1))

  V1=torch.from_numpy(V1)
  V0=torch.from_numpy(V0)

  return V1, V0

# Function to define the indicator function I

def I(sample, label, V1, V0):
  if label==1:
    return 1
  if LA.norm(sample-V1)<(LA.norm(sample-V0)):
    return 1

  return 0

# Function to generate training labels

def get_predicted_labels(data, target, V1, V0):
  n=target.shape[0]
  pred_label=np.zeros(n, dtype=np.float32)
  for i in range(n):
    pred_label[i]=I(data[i], target[i], V1, V0)

  return torch.from_numpy(pred_label)

# Function to create the set A for each session

def set_A(data, target, index):
  n=data.shape[0]
  m=n-1
  data=data.detach().cpu().numpy()
  target=target.detach().cpu().numpy()
  A=np.zeros([m,data.shape[1]], dtype=np.float32)
  Alabel=np.zeros(m, dtype=np.float32)
  j=0
  for i in range(n):
    if i != index:
      A[j]=data[i]
      Alabel[j]=target[i]
      j=j+1
  A=torch.from_numpy(A)
  Alabel=torch.from_numpy(Alabel)
  return A, Alabel

# Function to create set B for each session

def set_B(A, Alabel, label):
  n=A.shape[0]
  A=A.detach().cpu().numpy()
  Alabel=Alabel.detach().cpu().numpy()
  count=0
  for i in range(n):
    if Alabel[i]==label:
      count=count+1

  B=np.zeros([count,A.shape[1]], dtype=np.float32)
  Blabel=np.zeros(count, dtype=np.float32)

  count=0
  for i in range(n):
    if Alabel[i]==label:
      B[count]=A[i]
      Blabel[count]=Alabel[i]
      count=count+1

  B=torch.from_numpy(B)
  Blabel=torch.from_numpy(Blabel)
  return B, Blabel

# Pair debiased supervised contrastive loss function

def loss_pair(query, A, B, index, alpha=1):
  cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
  numerator=torch.exp((cos(query.view(1,features),B[index,:].view(1,features)))/alpha)
  denom=0
  for k in range(A.shape[0]):
    denom=denom+torch.exp((cos(query.view(1,features),A[k,:].view(1,features)))/alpha)

  score=-torch.log(torch.div(numerator,denom))

  return score

# Function to implement debiased supervised contrastive loss

def contrastive_loss(values, target, temp_label):
  V1,V0 = get_distribution_centers(values, target)
  pred_label = get_predicted_labels(values, target, V1, V0)
  batch_loss=0
  for i in range(values.shape[0]):
    flag = target[i]+temp_label[i]
    if flag!= 4:
      query=values[i]
      A, Alabel = set_A(values, pred_label, i)
      B, Blabel = set_B(A, Alabel, pred_label[i])
      loss=0
      for j in range(B.shape[0]):
        loss=loss+loss_pair(query, A, B, j)
      if B.shape[0] > 0:
        loss=loss/B.shape[0]
      batch_loss=batch_loss+loss

  return batch_loss

# Function to create the training batch

def make_dataset(act_file, session_count):
  dataset=np.zeros((session_count, seq_length, features))
  session_label_old= []
  temp_label_old= []
  session_number=0
  Lines = act_file.readlines()
  for line in Lines:
    acts, temp_label, label = line.split(';')
    temp_label_old.append(temp_label)
    session_label_old.append(label.strip('\n'))
    sequence_number=0
    for act in acts.split(','):
      if sequence_number<seq_length:
        x=word2Vec_model.wv.get_vector(act.lower())
        for i in range(features):
          dataset[session_number][sequence_number][i]=x[i]
        sequence_number=sequence_number+1
    session_number=session_number+1

  temp_label=np.array(temp_label_old)
  temp_label= temp_label.astype(np.float32)
  session_label=np.array(session_label_old)
  session_label=session_label.astype(np.float32)
  dataset=dataset.astype(np.float32)
  return dataset, temp_label, session_label

# Function to train the LSTM

def train_model(dataloader,net,optimizer, temp_label):
  for batch_idx, (data, targets) in enumerate(tqdm(dataloader)):
    optimizer.zero_grad()
    values, _ = net(data)
    loss = contrastive_loss(values, targets, temp_label)
    loss.backward()
    optimizer.step()

  return loss

# Function to generate the training batches and perfrom LSTM training for all the batches

def train_dataset(seq_length,features, model,optimizer,batch_size):
  session_count=batch_size
  loss_plot = []
  for epoch in range(num_epochs):
    for root, dirs, files in os.walk("training_set"):
      for filename in files:
        with open(os.path.join(root,filename), 'r', encoding='utf8') as act_file:
          data_vector, temp_label, label = make_dataset(act_file, session_count)
          temp_label=torch.from_numpy(temp_label)
          dataset=Seq_Dataset(data_vector,label)
          dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=len(dataset))
          loss = train_model(dataloader,model,optimizer, temp_label)
    loss_plot.append(loss)




  return loss, loss_plot

# Execute the training loop and store the model

loss, loss_plot = train_dataset(seq_length, features, net, optimizer,batch_size)
save_path = './network.pth'
torch.save(net.state_dict(), save_path)
word2Vec_model.save("word2vec.model")

