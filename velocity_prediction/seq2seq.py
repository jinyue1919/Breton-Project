# %%imports
import os
import pandas as pd
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from torch import Tensor
from torch import optim
import torch.nn.functional as F
import random
import math
import time
import plotly.graph_objects as go
import plotly.offline as pyo
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import matplotlib.pyplot as plt
import seaborn as sns
from fastai.basic_data import DataBunch# Set notebook mode to work in offline
from fastai.basic_train import Learner
from fastai import *
import fastai.train
import fastai.basic_train
import fastai.basic_data
from functools import partial
from fastai.callbacks import EarlyStoppingCallback, SaveModelCallback, tensorboard, ActivationStats
from pathlib import Path
from collections import OrderedDict


# %% define model
# process the sequence of inputs, return all hidden states and final layer outputs
# process the sequence of inputs, return all layers' hidden states and final layer outputs
class Encoder(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, num_layers, num_embedding=None, embedding_dim=None,
                 classification=False, rnn=nn.GRU, dropout=0., bidirectional=False):
        super(Encoder, self).__init__()
        self.classification = classification
        self.rnn = rnn(input_size, hidden_size, num_layers)
        
        if self.classification == True:
            self.embedding = nn.Embedding(num_embedding, embedding_dim)
            assert embedding_dim == input_size, 'Embedding dimension must equal to rnn input_size!'
            
    # process the complete input sequence, yield last layer outputs and hidden states
    def forward(self, input):
        
        # input: (seq_len, batch_size, n_features)
        
        if self.classification == False:
            if input.ndim == 2:
                # add a feature dimension for rnn calculation
                input = input.unsqueeze(-1)

            # input:(seq_len, batch_size, 1)

            outputs, hidden_states = self.rnn(input)

            # outputs are from the last layer: (seq_len, batch_size, hidden_size)

            # hidden_states
            # for lstm:
            # hidden_states are stacked and collected into a tuple: (hidden, cell), where
            # hidden and cell: (num_layers * num_directions, batch_size, hidden_size), num_directions = 2 if bidirectional = True else 1
            # for gru:
            # hidde_states: (num_layers * num_directions, batch_size, hidden_size)
        else:
            embedded = self.embedding(input)  # (seq_len, batch_size, embedding_dim)
            outputs, hidden_states = self.rnn(embedded)
            
        return outputs, hidden_states

# Use previous decoder hidden state and encoder outputs to calculate attention weights
class Attention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super().__init__()
        
        self.attn = nn.Linear(enc_hidden_size + dec_hidden_size, dec_hidden_size)
        self.v = nn.Linear(dec_hidden_size, 1, bias = False)
        
    def forward(self, dec_hidden, enc_outputs):
        
        # dec_hidden is final layer outputs from previous time step : (batch_size, hidden_size)
        # enc_outputs: (seq_len, batch_size, hidden_size)
        
        seq_len = enc_outputs.shape[0]
        batch_size = enc_outputs.shape[1]
        
        # repeat decoder previous hidden state {seq_len} times
        dec_hidden = dec_hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, dec_hidden_size)

        # switch batch and seq_len dimension
        enc_outputs = enc_outputs.permute(1, 0, 2)  # (batch_size, seq_len, enc_hidden_size)
        
        energy = torch.tanh(self.attn(torch.cat((dec_hidden, enc_outputs), dim=2)))  # (batch_size, seq_len, dec_hidden_size)
            
        attention = self.v(energy).squeeze(2)  # (batch_size, seq_len)
                      
        return F.softmax(attention, dim=1)# calculate a single time step

# Calculate a single time step output
class Decoder(nn.Module):
    def __init__(self, trg_len, input_size, enc_hidden_size, dec_hidden_size, output_size, num_layers, attention, rnn, dropout=None):
        super().__init__()
        
        self.trg_len = trg_len
        self.output_size = output_size
        self.attention = attention
        self.dropout = dropout
        
        self.rnn = rnn(enc_hidden_size + input_size, dec_hidden_size, num_layers=num_layers)
        self.fc_out = nn.Linear(input_size + enc_hidden_size + dec_hidden_size, output_size)

        
    def forward(self, input, dec_outputs, enc_outputs, hidden, require_attention=True):
        
        # input: (batch_size, output_size) -> only calculate a single time step. output_size is 1 here (predicted velocity)
        # dec_outputs: (batch_size, dec_hidden_size) -> decoder's last layer outputs from previous time step
        # enc_outputs: (seq_len, batch_size, enc_hidden_size)
        # hidden is: (hidden should be provided every time otherwise pytorch will set them to all zeros) 
        # encoder all layers' hidden states in last time step: (num_layers, batch_size, enc_hidden_size)
        # and decoder previous hidden states in other cases: (num_layers, batch_size, dec_hidden_size)
        
        if input.ndim == 1:
            # input has only one feature: (batch_size)
            # add a {seq_len} and {feature} dimension
            input = input.reshape(1, -1, 1)
        elif input.ndim == 2:
            # input has more than one feature
            # add a {seq_len} dimension
            input = input.unsqueeze(0)
        
        # input: (1, batch_size, n_features)
        
        if require_attention == True:
            
            # use decoder last layer outputs from previous time step as query for attention weights
            # calculate attention weights and add a dimension for torch.bmm calculating
            attention = self.attention(dec_outputs, enc_outputs).unsqueeze(1)

            # attention: (batch_size, 1, seq_len)

            enc_outputs = enc_outputs.permute(1, 0, 2)

            # enc_outputs: (batch_size, seq_len, enc_hidden_size)

            weighted_context = torch.bmm(attention, enc_outputs)

            # weighted_context: (batch_size, 1, enc_hidden_size)

            weighted_context = weighted_context.permute(1, 0, 2)

            # concatenate the feature dimension
            rnn_input = torch.cat((input, weighted_context), dim=2)

            # rnn_input: (1, batch_size, input_size + enc_hidden_size)
        
        elif require_attention == False:
            
            # context vector is encoder last layer hidden state (output) at last time step
            # add a {seq_len} dimension
            context = enc_outputs[-1].unsqueeze(0)
            
            # concatenate context vector and decoder last step prediction
            rnn_input = torch.cat((input, context), dim=2)
        
        outputs, hidden_states = self.rnn(rnn_input, hidden)
        
        # outputs: (1, batch_size, dec_hidden_size)
        # hidden_states: (num_layers, batch_size, dec_hidden_size)
        
        # because Decoder only calculates one time step, so seq_len == 1
        input = input.squeeze(0)
        outputs = outputs.squeeze(0)
        
        if require_attention == True:
            
            weighted_context = weighted_context.squeeze(0)

            # torch.cat((input, weighted_context, outputs)): (1, batch_size, 1 + enc_hidden_size + dec_hidden_size)

            prediction = self.fc_out(torch.cat((input, weighted_context, outputs), dim=1))

            # prediction: (batch_size, output_size)
        
        elif require_attention == False:
            context = context.squeeze(0)
            prediction = self.fc_out(torch.cat((input, context, outputs), dim=1))
            
        # return prediction for mse loss calculation and decoder all layers' hidden states for next time step calculation
        return prediction, outputs, hidden_states

# encapsulate Encoder, Attention and Decoder into Seq2SeqWithAttention
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, require_attention=True, teacher_forcing_ratio=0):
        super().__init__()

        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.encoder = encoder
        self.decoder = decoder
        self.require_attention = require_attention

    def forward(self, src):

        if src.ndim == 2:
            # src: (batch_size, src_len)
            src = src.t().unsqueeze(-1)
        elif src.ndim == 3:
            # src: (batch_size, src_len, n_features)
            src = src.permute(1, 0, 2)

        src_len = src.shape[0]
        batch_size = src.shape[1]
        trg_len = self.decoder.trg_len

        # tensor to store all decoder predictions
        dec_predictions = torch.zeros(trg_len, batch_size, self.decoder.output_size)

        enc_outputs, enc_hidden_states = self.encoder(src)

        # encoder last layer hidden states
        # if isinstance(self.encoder.rnn, nn.LSTM):
        #     h, _ = enc_hidden_states
        #     hidden = h[-1]
        # elif isinstance(self.encoder.rnn, nn.GRU):
        #     hidden = enc_hidden_states[-1] # (batch_size, enc_hidden_size)

        # first input of decoder is last step of src's velocity dimension
        input = src[-1, :, 0]
        
        # decoder initial hidden is encoder final hidden 
        hidden = enc_hidden_states
        
        # decoder initial 'last layer outputs' is encoder last layer's outputs from final time step
        output = enc_outputs[-1]
        
        for t in range(trg_len):
            
            prediction, output, hidden = self.decoder(input, output, enc_outputs, hidden, require_attention=self.require_attention)
            
            # prediction: (batch_size, decoder.output_size)

            # collect predictions 
            dec_predictions[t] = prediction

                # teacher_force = (random.random() < self.teacher_forcing_ratio) if self.teacher_forcing_ratio > 0 else False
                # input = trg[t] if teacher_force else output
            
            # set next time step's input as previous prediction
            input = prediction

        # transpose to (batch_size, trg_len) for loss calculation
        return dec_predictions.squeeze(-1).t()

# %% utility functions 
# split a univariate sequence into samples
def insertSOSToken(sequence, SOS):
    sequence.insert(0, SOS)
    return sequence
    
def split_sequences(sequence, n_steps_in, n_steps_out, SOS=0, velocity=False):
    seq_x_final, seq_y_final = [], []
    if sequence.ndim == 1:
        for i in range(len(sequence) - n_steps_in - n_steps_out + 1):
            in_end_idx = i + n_steps_in
            out_end_idx = in_end_idx + n_steps_out
            seq_x, seq_y = sequence[i:in_end_idx], sequence[in_end_idx:out_end_idx]
            if velocity == True:
                if np.count_nonzero(seq_x>85.) == 0 and np.count_nonzero(seq_y>85.) == 0 and np.unique(seq_x).size != 1 and np.unique(seq_y).size != 1:
    #                 seq_y = insertSOSToken(seq_y, SOS)
                    seq_x_final.append(seq_x)
                    seq_y_final.append(seq_y)
            else:
                seq_x_final.append(seq_x)
                seq_y_final.append(seq_y)
    elif sequence.ndim == 2:
        for i in range(sequence.shape[1] - n_steps_in - n_steps_out + 1):
            in_end_idx = i + n_steps_in
            out_end_idx = in_end_idx + n_steps_out
            if np.isclose(np.isclose(sequence[0, i:i+n_steps_in+n_steps_out], np.zeros(n_steps_in+n_steps_out)).astype('float32').sum(), 0):
                seq_x, seq_y = sequence[:, i:in_end_idx], sequence[:, in_end_idx:out_end_idx]
                if velocity == True:
                    # seq_x[0] is velocity 
                    if np.count_nonzero(seq_x[0] > 85) == 0 and np.count_nonzero(seq_y[0] > 85) == 0 and np.unique(seq_x[0]).size != 1 and np.unique(seq_y[0]).size != 1:
        #                 seq_y = insertSOSToken(seq_y, SOS)
                        seq_x_final.append(seq_x)
                        seq_y_final.append(seq_y)
                else:
                    seq_x_final.append(seq_x)
                    seq_y_final.append(seq_y)
    return np.array(seq_x_final), np.array(seq_y_final)

def mean_absolute_percentage_error(y_true, y_hat):
    X = []
    Y = []
    for x, y in zip(y_true, y_hat):
        if (x != 0) and (y != 0):
            X.append(x)
            Y.append(y)
    X = np.array(X)
    Y = np.array(Y)
    return np.mean(np.abs((X - Y) / X)) * 100

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def extractValidSeq(seq, min_len=30*5):
    '''
    return an array of valid indexes
    '''
    # gearFlag = {0: null, 1: D, 2: N, 3: R, 4: L, 5: P, 6: M}
    validIndexesTemp = np.hstack([np.nonzero(seq==1), np.nonzero(seq==4), np.nonzero(seq==6)]).flatten()
    validIndexesTemp = consecutive(validIndexesTemp)
    validIndexes = []
    for arr in validIndexesTemp:
        if arr.size >= min_len:
            validIndexes.append(arr)
    return validIndexes

def moving_average(x, window_len=3):
    # x: np array
    w = np.ones(window_len)
    if x.ndim == 1:
        s = np.r_[x[0], x, x[-1]]  # padding to make the length stay the same
        x_filtered = np.convolve(w/w.sum(), s, mode='valid')
    elif x.ndim > 1:
        x_filtered = np.zeros(x.shape)
        for i, xx in enumerate(x):
            s = np.r_[xx[0], xx, xx[-1]]
            x_filtered[i] = np.convolve(w/w.sum(), s, mode='valid')
    return x_filtered

def longest_seq_idx(x):
    max = 0
    idx = 0
    for i, arr in enumerate(x):
        if arr.size > max:
            max = arr.size
            idx = i
    return idx

# %% data preparation
n_steps_in = 15
n_steps_out = 10
n_features = 1
batch_size = 64

vel_CAN_cycle = 0.2  # unit: s
vel_count_per_second = int(1 / vel_CAN_cycle)
dataFolder = 'BretonDataTest'
velocity_files = os.listdir(dataFolder)
velocity_files.sort()
vel_seq_x = []  # Final list that contains all inputs
vel_seq_y = []  # Labels
data_seq_x = []
data_seq_y = []

classification = True
SOS = 0

for file in velocity_files:
    fileName = os.path.join(dataFolder, file)
    df = pd.read_csv(fileName, header=None, sep='\s+', names=['vel', 'acc', 'brake', 'gear', 'gearFlag'],
        dtype={'vel': np.float32, 'acc': np.float32, 'brake': np.float32, 'gear': np.float32, 'gearFlag': np.float32})
    velocity = df['vel'].values
    gear = df['gear'].values
    gearFlag = df['gearFlag'].values
    acc = df['acc'].values
    assert velocity.size == acc.size
    
    valid_indexes = extractValidSeq(gearFlag)
    for arr in valid_indexes:
        vel_valid = velocity[arr]
        
        if n_features > 1:
            acc_valid = acc[arr]
            data_valid = np.vstack((vel_valid, acc_valid))
            data_seq_temp = np.array([data_valid[:,i:i + vel_count_per_second].mean(1) for i in range(0, vel_valid.size, vel_count_per_second) if (i + vel_count_per_second <= vel_valid.size)])
            data_seq_temp = data_seq_temp.transpose()
            data_seq_temp_filtered = moving_average(data_seq_temp)
            data_seq_x_temp, data_seq_y_temp = split_sequences(data_seq_temp_filtered, n_steps_in, n_steps_out, velocity=True)
            data_seq_x.append(data_seq_x_temp)
            data_seq_y.append(data_seq_y_temp)
            
        else:
            data_seq_temp = np.array([vel_valid[i:i + vel_count_per_second].mean() for i in range(0, vel_valid.size, vel_count_per_second) if (i + vel_count_per_second <= vel_valid.size)])
            data_seq_temp_filtered = moving_average(data_seq_temp)

    #     gear_list = [gear[i] for i in range(0, gear.size, vel_count_per_second) if (i + vel_count_per_second <= gear.size)]
    #     gearFlag_list = [gearFlag[i] for i in range(0, gearFlag.size, vel_count_per_second) if (i + vel_count_per_second <= gearFlag.size)]

            data_seq_x_temp, data_seq_y_temp = split_sequences(data_seq_temp_filtered, n_steps_in, n_steps_out, velocity=True)
            data_seq_x.append(data_seq_x_temp)
            data_seq_y.append(data_seq_y_temp)

while True:
    invalid = []
    for i, (x, y) in enumerate(zip(data_seq_x, data_seq_y)):
        if x.size == 0 or y.size == 0:
            invalid.append(i)
            data_seq_x.pop(i)
            data_seq_y.pop(i)
    if len(invalid) == 0:
        break

data_X = np.concatenate(data_seq_x)
data_Y = np.concatenate(data_seq_y)
assert data_X.shape[0] == data_Y.shape[0]
print(f'Total {data_X.shape[0]} samples')

# vel_X, vel_Y = generateSamples(vel_seq_x, vel_seq_y, n_steps_in, n_steps_out, classification=False)

# shuffle the datasets
randomIndexes = torch.randperm(data_X.shape[0])
data_X_shuffled = data_X[randomIndexes]
data_Y_shuffled = data_Y[randomIndexes]
assert np.allclose(data_X_shuffled[0], data_X[randomIndexes[0]])
data_X = data_X_shuffled
data_Y = data_Y_shuffled

# %% generate Pytorch datasets
class Datasets():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    
    def __len__(self):
        return self.x.shape[0]

# min-max scaling
if n_features == 1:
    data_X = data_X.astype('float32') / 85.
    data_Y = data_Y.astype('float32') / 85.

    x_train = data_X[:int(0.8 * data_X.shape[0])] 
    y_train = data_Y[:int(0.8 * data_Y.shape[0])]
    x_valid = data_X[int(0.8 * data_X.shape[0]):int(0.9 * data_X.shape[0])]
    y_valid = data_Y[int(0.8 * data_Y.shape[0]):int(0.9 * data_X.shape[0])]
    x_test = data_X[int(0.9 * data_X.shape[0]):]
    y_test = data_Y[int(0.9 * data_Y.shape[0]):]
elif n_features == 2:
    data_X = data_X.astype('float32').swapaxes(1, 2)
    # the target only contains velocity
    data_Y = data_Y.astype('float32').swapaxes(1, 2)[:, :, 0]
    data_X[:, :, 0] = data_X[:, :, 0] / 85
    data_Y = data_Y / 85.
    x_train = data_X[:int(0.8 * data_X.shape[0])]
    y_train = data_Y[:int(0.8 * data_Y.shape[0])]
    x_valid = data_X[int(0.8 * data_X.shape[0]):int(0.9 * data_X.shape[0])]
    y_valid = data_Y[int(0.8 * data_Y.shape[0]):int(0.9 * data_X.shape[0])]
    x_test = data_X[int(0.9 * data_X.shape[0]):]
    y_test = data_Y[int(0.9 * data_Y.shape[0]):]
assert x_train.shape[0] == y_train.shape[0]
assert x_valid.shape[0] == y_valid.shape[0]
assert x_test.shape[0] == y_test.shape[0]

train_ds, valid_ds, test_ds = Datasets(x_train, y_train), Datasets(x_valid, y_valid), Datasets(x_test, y_test)
assert len(train_ds) == x_train.shape[0]

train_dl = DataLoader(train_ds, batch_size=64, sampler=RandomSampler(train_ds))
valid_dl = DataLoader(valid_ds, batch_size=64, sampler=SequentialSampler(valid_ds))
test_dl = DataLoader(test_ds, batch_size=64, sampler=SequentialSampler(test_ds))

# %% training the model
# model parameters
batch_size = 64

enc_input_size = 1
dec_input_size = 1

enc_hidden_size = 32
dec_hidden_size = 32
enc_num_layers = 2
dec_num_layers = 2
dec_output_size = 1

BIDIRECTIONAL = False
rnn = nn.GRU
rnn = nn.LSTM

device = 'cpu'
        
enc = Encoder(batch_size, enc_input_size, enc_hidden_size, enc_num_layers, rnn=rnn)
attn = Attention(enc_hidden_size, dec_hidden_size)
dec = Decoder(n_steps_out, dec_input_size, enc_hidden_size, dec_hidden_size, dec_output_size, dec_num_layers, attention=attn, rnn=rnn)
model = Seq2Seq(enc, dec, require_attention=False).to(device)

# model = nn.Sequential(OrderedDict([
#     ('fc1', nn.Linear(n_steps_in, 64)),
#     ('relu', nn.ReLU()),
#     ('fc2', nn.Linear(64, 64)),
#     ('relu', nn.ReLU()),
#     ('fc3', nn.Linear(64, n_steps_out)),
#     ('relu', nn.ReLU())
# ]))

# print(model)

# %% model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

# %% use fastai library
class ArrayDataset():
    "Sample numpy array dataset"
    def __init__(self, x, y):
        self.x, self.y = x, y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]

train_ds, valid_ds = ArrayDataset(x_train, y_train), ArrayDataset(x_valid, y_valid)
data = DataBunch.create(train_ds, valid_ds, bs=64, num_workers=1)

# create learner
loss_func = nn.MSELoss()
# opt = partial(optim.SGD, lr=1e-8, momentum=0.9)
opt = optim.Adam
learn = Learner(data,
                model,
                opt_func=opt,
                loss_func=loss_func)#,
#                callback_fns=[lr_finder])

lr_find_epochs = 2
num_it = lr_find_epochs * len(data.train_dl)
learn.lr_find(start_lr=1e-8, end_lr=10, num_it=num_it, stop_div=True)


from fastai.callbacks import EarlyStoppingCallback, SaveModelCallback, tensorboard, ActivationStats
from pathlib import Path

EarlyStopping = EarlyStoppingCallback(learn, monitor='valid_loss', min_delta=1e-4, patience=10)
SaveBest = SaveModelCallback(learn, every='improvement', monitor='valid_loss', name='bestmodel')
project_id = '0427'
tboard_path = Path('data/tensorboard/' + project_id)
Tensorboard = tensorboard.LearnerTensorboardWriter(learn, base_dir=tboard_path, name='run1')
ActiStats = ActivationStats(learn)
min_lr = 1e-4
max_lr = 1e-3
div_factor = max_lr / min_lr
learn.fit_one_cycle(10, max_lr=max_lr, div_factor=div_factor, callbacks=[EarlyStopping, SaveBest])
model_static_dict = torch.save(model.state_dict(), '0706seq2seq')

# %%
