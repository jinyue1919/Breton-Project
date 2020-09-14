#%% imports
import os
import time
import progressbar
from util_vel_pred import *
from energy_opt_v1 import *
from energy_opt_v2 import *
import pickle
import math
import pdb

#%% utility functions
def find(list, value):
	return [i for i, x in enumerate(list) if x == value]

#%% model parameters
n_steps_in = 15
n_steps_out = 10
n_features = 2
batch_size = 64

enc_input_size = 1
dec_input_size = 1
enc_hidden_size = 32
dec_hidden_size = 32
enc_num_layers = 2
dec_num_layers = 2
dec_output_size = 1

BIDIRECTIONAL = False
rnn = nn.LSTM

device = 'cpu'
		
enc = Encoder(batch_size, enc_input_size, enc_hidden_size, enc_num_layers, rnn=rnn)
attn = Attention(enc_hidden_size, dec_hidden_size)
dec = Decoder(n_steps_out, dec_input_size, enc_hidden_size, dec_hidden_size, dec_output_size, dec_num_layers, attention=attn, rnn=rnn)
model = Seq2Seq(enc, dec, require_attention=False).to(device)
# model.load_state_dict(torch.load('/Users/tujiayu/Dev/BretonProject/Velocity Prediction/models/0427seq2seqWithAttention-layers2-units32-wd0.001-lr1e-5-2e-4.pth'))
# model.load_state_dict(torch.load('models/0427seq2seqNoAttention-layers2-units32-wd0.001-lr1e-5-3e-4.pth'))
# model.load_state_dict(torch.load('models/0705seq2seqNoAttention-no-acc-layers2-units32-wd0.001-lr12e-5-2e-4-minmax.pth'))
model.load_state_dict(torch.load('models/0706seq2seq'))

vel_CAN_cycle = 0.2  # unit: s
vel_count_per_second = int(1 / vel_CAN_cycle)
dataFolder = 'BretonDataTest'
# velocity_files = os.listdir(dataFolder)
# velocity_files.sort()
# for file in velocity_files:
file = '0618_1.csv'
fileName = os.path.join(dataFolder, file)
df = pd.read_csv(fileName, header=None, sep='\s+', names=['vel', 'acc', 'brake', 'gear', 'gearFlag'],
	dtype={'vel': np.float32, 'acc': np.float32, 'brake': np.float32, 'gear': np.float32, 'gearFlag': np.float32})
velocity = df['vel'].values
gear = df['gear'].values
gearFlag = df['gearFlag'].values
acc = df['acc'].values
brake = df['brake'].values

assert velocity.size == gear.size == acc.size == brake.size

valid_indexes = extractValidSeq(gearFlag)
sorted_indexes = sorted(valid_indexes, key=lambda x: x.size, reverse=True)

for count, idxes in enumerate(sorted_indexes):
    if len(idxes) > 30 * 60 * 5:  # longer than 30 mins
        V = velocity[sorted_indexes[count]]
        A = acc[sorted_indexes[count]]
        B = brake[sorted_indexes[count]]
        G = gear[sorted_indexes[count]]

        v = np.array([V[i:i + vel_count_per_second].mean() for i in range(0, V.size, vel_count_per_second) if i + vel_count_per_second <= V.size])

        # if an average is used, there may be cases where both the accelerator and brake are greater than 0
        a = np.array([A[i] for i in range(0, A.size, vel_count_per_second) if i + vel_count_per_second <= A.size])
        b = np.array([B[i] for i in range(0, B.size, vel_count_per_second) if i + vel_count_per_second <= B.size])
        g = np.array([G[i] for i in range(0, G.size, vel_count_per_second) if i + vel_count_per_second <= G.size])

        data = np.vstack((v, a)) 
        data = moving_average(data)  # (n_features, )
        v = moving_average(v)


        mode = 'min-max'
        assert data[0].shape == g.shape == b.shape == a.shape
        
        data = v
        final_data = np.zeros((v.size, 1 + 1 + n_steps_out))
        for i in range(n_steps_in):
            final_data[i, 0] = v[i]
            final_data[i, 1] = g[i]

        for i in progressbar.progressbar(range(n_steps_in, v.size - n_steps_out)):
            data_history = data[i - n_steps_in:i].copy()
            data_history = data_history / 85.
            vel_pred = predict(data_history, model) * 85           
            vel_pred = vel_pred.detach().numpy()
            final_data[i, 0] = v[i]
            final_data[i, 1] = g[i]
            final_data[i, 2:] = vel_pred.squeeze()

        np.savetxt(f'breton_data_{count}.csv', final_data, fmt="%.2f", delimiter=',')
