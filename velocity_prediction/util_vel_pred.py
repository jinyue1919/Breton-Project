from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Dataset():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)

def get_dls(train_ds, valid_ds, bs, **kwargs):
    return (DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
            DataLoader(valid_ds, batch_size=bs, **kwargs))

class DataBunch():
    def __init__(self, train_dl, valid_dl, c=None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.c = c
    
    # make train_ds attribute
    @property
    def train_ds(self):
        return self.train_dl.dataset

    # make valid_ds attribute
    @property
    def valid_ds(self):
        return self.valid_dl.dataset

class Learner():
    def __init__(self, model, opt, loss_func, data):  # DataBunch for data
        self.model,self.opt,self.loss_func,self.data = model,opt,loss_func,data

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
                if np.count_nonzero(seq_x==0) == 0 and np.count_nonzero(seq_y==0) == 0 and np.count_nonzero(seq_x>85.) == 0 and \
                    np.count_nonzero(seq_y>85.) == 0 and np.unique(seq_x).size != 1 and np.unique(seq_y).size != 1:
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
                    if np.count_nonzero(seq_x[0] == 0) < 1 and np.count_nonzero(seq_y[0] == 0) < 1 and np.count_nonzero(seq_x[0] > 85) == 0 and \
                    np.count_nonzero(seq_y[0] > 85) == 0 and np.unique(seq_x[0]).size != 1 and np.unique(seq_y[0]).size != 1:
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

def RMSEerror(x, y):
    # x, y: (batch_size, element_num)
    if x.ndim == 1:
        x = x[None,]
    return (((x - y)**2).mean(1)**0.5).mean()

def RMSEOfDatasets(model, test_dl):
    test_rmse = []
    loss_func = nn.MSELoss()
    for xb, yb in test_dl:
        ori = yb * 85
        pred = model(xb) * 85
        rmse = RMSEerror(ori, pred)
        test_rmse.append(rmse.item())
    return sum(test_rmse) / len(test_rmse)

def MAEerror(x, y):
    return (x - y).abs().mean()

def MAPEerror(x, y):
    return ((x - y).abs() / x).mean() *1000

def predict(xb, model):
    # xb's shape should be (batch_size, seq_len, n_features)    
    if xb.ndim == 2:  # suitable for both ndarray and Tensor
        # add a {batch_size} dim
        xb = xb[None, ]
    elif xb.ndim == 1:
        xb = xb[None, :, None]
    if not isinstance(xb, torch.Tensor):
        xb = torch.Tensor(xb)
    return model(xb)

v_mean = None
v_std = None
a_mean = None
a_std = None

print('Import succeed!')