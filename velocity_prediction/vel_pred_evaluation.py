#%% imports
from energy_opt_v2 import *
from util_vel_pred import *
import pandas
import os
import time
import progressbar

#%% utility functions
def find(list, value):
    return [i for i, x in enumerate(list) if x == value]

#%% parameters
n_steps_in = 15
n_steps_out = 10
n_features = 2
batch_size = 64

vel_CAN_cycle = 0.2  # unit: s
vel_count_per_second = int(1 / vel_CAN_cycle)
dataFolder = 'BretonDataTest'
velocity_files = os.listdir(dataFolder)
velocity_files.sort()

#%% extract valid sequences from files
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

#%% model parameters
batch_size = 64

enc_input_size = 2
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
# model.load_state_dict(torch.load('/Users/tujiayu/Dev/BretonProject/Velocity Prediction/models/0427seq2seqWithAttention-layers2-units32-wd0.001-lr1e-5-2e-4.pth'))
model.load_state_dict(torch.load('models/0427seq2seqNoAttention-layers2-units32-wd0.001-lr1e-5-3e-4.pth'))

#%%
mode = 'min-max'
rmse_final = []
outlier_rmse_final = []
outlier_vel_final = []

for idx in range(5):

    v = velocity[sorted_indexes[idx]]
    a = acc[sorted_indexes[idx]]
    data_raw = np.vstack((v, a))
    data_raw = np.array([data_raw[:,i:i + vel_count_per_second].mean(1) for i in range(0, v.size, vel_count_per_second) if (i + vel_count_per_second <= v.size)])
    data_raw = data_raw.transpose()
    data = moving_average(data_raw)

    rmse = []

    for i in range(n_steps_in, data.shape[1]-n_steps_out): 
        # i is current step
        data_history = data[:, i-n_steps_in:i].copy()
        # vel_history, acc_history = data_history[0], data_history[1]
        if n_features == 1:
            print(f'Need to modify')
        elif n_features == 2:
            if mode == 'min-max':
                # min-max scaling
                data_history[0] = data_history[0] / 85.
                assert list(data_history.shape) == [n_features, n_steps_in]
                vel_pred = predict(data_history.transpose(), model) * 85
            elif mode == 'std':
                data_history[0] = (data_history[0] - v_mean) / v_std
                data_history[1] = (data_history[1] - a_mean) / a_std
                vel_pred = predict(data_history.transpose(), model) * v_std + v_mean
            vel_pred = vel_pred.detach().numpy()
            vel_real = data[0, i:i+n_steps_out]
            rmse_temp = RMSEerror(vel_pred, vel_real) 
            rmse.append(rmse_temp)

    r = plt.boxplot(rmse)
    rmse_outlier = r["fliers"][0].get_data()[1].min()
    outlier_rmse = []
    outlier_vel = []
    
    for i in range(n_steps_in, data.shape[1]-n_steps_out): 
        # i is current step
        data_history = data[:, i-n_steps_in:i].copy()
        # vel_history, acc_history = data_history[0], data_history[1]
        if n_features == 1:
            print(f'Need to modify')
        elif n_features == 2:
            if mode == 'min-max':
                # min-max scaling
                data_history[0] = data_history[0] / 85.
                assert list(data_history.shape) == [n_features, n_steps_in]
                vel_pred = predict(data_history.transpose(), model) * 85
            elif mode == 'std':
                data_history[0] = (data_history[0] - v_mean) / v_std
                data_history[1] = (data_history[1] - a_mean) / a_std
                vel_pred = predict(data_history.transpose(), model) * v_std + v_mean
            vel_pred = vel_pred.detach().numpy()
            vel_real = data[0, i:i+n_steps_out]
            rmse_temp = RMSEerror(vel_pred, vel_real) 
            rmse.append(rmse_temp)
        if rmse_temp > rmse_outlier:
            outlier_vel.append(vel_real.mean())
            outlier_rmse.append(rmse_temp)
    
    rmse_final.append(rmse)
    outlier_vel_final.append(outlier_vel)
    outlier_rmse_final.append(outlier_rmse)
