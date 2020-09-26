#%% imports
import os
import time
import progressbar
from util_vel_pred import *
from energy_opt_v1 import *
from energy_opt_v2 import *
from energy_opt_v3 import *
import pickle
from fuzzy_w_modify import *
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
model.load_state_dict(torch.load('models/0705seq2seqNoAttention-no-acc-layers2-units32-wd0.001-lr12e-5-2e-4-minmax.pth'))

#%% extract valid sequences from files
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

#%%
idx = 0  # the longest
V = velocity[sorted_indexes[idx]]
A = acc[sorted_indexes[idx]]
B = brake[sorted_indexes[idx]]
G = gear[sorted_indexes[idx]]

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

# data, a, b, g should not be changed, use copy of them if want to access a slice of them

# %%
opt_results_v1 = {}


################ 存储标志位 ################
opt_results_v1['flag'] = []

################ 存储纯优化的数据 ################
opt_results_v1['vel_opt'] = [] 
opt_results_v1['torque_opt'] = []  
opt_results_v1['gear_opt'] = []
opt_results_v1['motor_eff_opt'] = []
opt_results_v1['energy_opt'] = []
opt_results_v1['torque_wheel_opt'] = []
opt_results_v1['motor_speed_opt'] = []

################ 存储原始数据 ################
opt_results_v1['vel_dmd'] = []
opt_results_v1['torque_dmd'] = []
opt_results_v1['gear_dmd'] = []
opt_results_v1['motor_eff_dmd'] = []

opt_results_v1['torque_wheel_dmd'] = []
opt_results_v1['motor_speed_dmd'] = []

################ 存储实际执行的数据（兼顾驾驶意图的） ################
opt_results_v1['vel_ctl'] = []
opt_results_v1['torque_ctl'] = []
opt_results_v1['gear_ctl'] = []
opt_results_v1['motor_eff_ctl'] = []
opt_results_v1['energy_ctl'] = []
opt_results_v1['torque_wheel_ctl'] = []
opt_results_v1['motor_speed_ctl'] = []

################ 存储权重及权重变化率 ################
opt_results_v1['w1'] = []
opt_results_v1['W1'] = []

################ 一些并不知道为什么要存的东西 ################
opt_results_v1['mean_vel_past'] = []
opt_results_v1['mean_vel_pred'] = []
opt_results_v1['mean_vel_real'] = []
opt_results_v1['mean_vel_opt'] = []
opt_results_v1['invalid_indexes'] = []
opt_results_v1['vel_real'] = []
opt_results_v1['vel_pred'] = []
opt_results_v1['vel_min'] = []
opt_results_v1['vel_max'] = []
opt_results_v1['if_opt'] = []
opt_results_v1['abnormal_vel'] = []
opt_results_v1['vel_mean'] = []
opt_results_v1['Tm_wheel_diff_ctl'] = []
opt_results_v1['drive_mode'] = []
opt_results_v1['flag_torque_dmd'] = []
opt_results_v1['flag_torque_ctl'] = []
opt_results_v1['flag_torque_opt'] = []
opt_results_v1['flag_motor_speed_dmd'] = []
opt_results_v1['flag_motor_speed_ctl'] = []
opt_results_v1['flag_motor_speed_opt'] = []

vel_only = True
if vel_only == True:
	data = v

#%% segment optimization with dp
# backward simulation does not need to determine the vehicle state (drive, brake, slide, etc.)
# for i in progressbar.progressbar(range(n_steps_in, 100)):#data.shape[1] - n_steps_out)):
plt.style.use('ggplot')
count = 0
W1_list = []
w1_list = [] #预测里程和实际里程的偏差
W1 = 0.5
w1 = 0.5

torque_ori_list = []
torque_ctl_list = []

for i in progressbar.progressbar(range(n_steps_in, v.size - n_steps_out)):
# for i in progressbar.progressbar(range(n_steps_in, 500)):
	if i == n_steps_in:
		# Calculate original velocity, torque, motor_eff, energy
		vel_current = v[i - 1]
		vel_next = v[i]
		gear_next = gears_in_use[int(g[i] - 1)]

		_, torque, _ = torque_calc(vel_current / 3.6 ,vel_next / 3.6, gear_next)	

		if vel_only == True:
			vel_seq = data[i - n_steps_in:i + n_steps_out].copy()  # use copy of data to avoid changes
		else:
			vel_seq = data[0, i - n_steps_in:i + n_steps_out].copy()
		
		data_history = data[i - n_steps_in:i].copy()  # need to update every iteration
	else:
		vel_current = vel_next
		vel_next = v[i]
		gear_ctl = gear_next  # last step 
		gear_next = gears_in_use[int(g[i] - 1)]
		
		_, torque, _ = torque_calc(vel_current / 3.6 ,vel_next / 3.6, gear_next)	

		if vel_only == True:
			vel_seq = data[i - n_steps_in:i + n_steps_out].copy()
			vel_seq[n_steps_in-1] = vel_current
		else:
			vel_seq = data[0, i - n_steps_in:i + n_steps_out].copy()
			vel_seq[0, n_steps_in-1] = vel_current

	# optimization module is activated only when the car is in drive mode
	if torque >= 0:
		# the past {n_steps_in} seconds of velocity should all > 0
		if np.isclose(np.isclose(vel_seq, np.zeros(n_steps_in + n_steps_out)).astype('float32').sum(), 0):
			# i is current step
			if vel_only == True:
				# data_history = data[i - n_steps_in:i].copy()
				# data_history[-1] = vel_current  # update velocity
				vel_history = data[i - n_steps_in:i].copy()
			else:
				# data_history = data[:, i - n_steps_in:i].copy()  
				# data_history[0, -1] = vel_current
				vel_history = data[:, i - n_steps_in:i].copy()
			vel_history[-1] = vel_current

			if mode == 'min-max':
				if vel_only == True:
					# print(data_history)
					vel_pred = predict(data_history / 85, model) * 85
					# print(vel_pred)
				else:
					vel_pred = predict(data_history / 85, model) * 85
			elif mode == 'std':
				if vel_only == True:
					data_history[0] = (data_history[0] - v_mean) / v_std
					data_history[1] = (data_history[1] - a_mean) / a_std
					vel_pred = predict(data_history.transpose(), model) * v_std + v_mean
				else:
					data_history = (data_history - v_mean) / v_std

			vel_pred = vel_pred.detach().numpy()
			if vel_only == True:
				vel_real = data[i:i + n_steps_out].copy()
			else:
				vel_real = data[0, i:i + n_steps_out].copy()

			gear_pre = int(g[i - 1])
			
			if np.count_nonzero(vel_pred <= 0) > 0:
				vel_mean = distance_calc(vel_current/3.6, vel_next/3.6)

				gear_ctl = gear_next
				gear_next = gear_ctl
				# update data_history
				data_history = np.hstack([data_history[1:], [vel_next]])

				opt_results_v1['abnormal_vel'].append(i)
				opt_results_v1['flag'].append(-1)
				opt_results_v1['vel_opt'].append(-1)
				opt_results_v1['gear_opt'].append(-1)
				opt_results_v1['torque_opt'].append(-1)
				opt_results_v1['motor_eff_opt'].append(-1)
				opt_results_v1['mean_vel_past'].append(vel_history.mean() * 85)
				opt_results_v1['mean_vel_pred'].append(vel_pred.mean())
				opt_results_v1['mean_vel_real'].append(vel_real.mean())	
				opt_results_v1['mean_vel_opt'].append(-1)
				opt_results_v1['vel_real'].append(vel_real)
				opt_results_v1['vel_pred'].append(vel_pred)
				opt_results_v1['vel_min'].append(-1)
				opt_results_v1['vel_max'].append(-1)
				opt_results_v1['energy_opt'].append(-1)
				opt_results_v1['Tm_wheel_diff_opt'].append(-1)
				opt_results_v1['flag_motor_speed_opt'].append(-1)
				opt_results_v1['flag_torque_opt'].append(-1)
				#TODO gear_next ??
				# select a gear with the smallest energy 
				energy, torque_seq, motor_eff_seq, flag, motor_speed = energy_and_motor_eff_calc(np.array([vel_current, vel_next])/3.6, np.array([gear_next]), per_meter=False)
				
				opt_results_v1['energy_dmd'].append(energy)
				opt_results_v1['energy_ctl'].append(energy)
				opt_results_v1['if_opt'].append(0)
				opt_results_v1['Tm_wheel_diff_ctl'].append(0)
				opt_results_v1['vel_mean'].append(vel_mean)
				opt_results_v1['vel_ctl'].append(vel_next)
				opt_results_v1['vel_dmd'].append(v[i])
				opt_results_v1['torque_ctl'].append(torque_seq[0])
				opt_results_v1['gear_ctl'].append(g[i])
				opt_results_v1['motor_eff_ctl'].append(motor_eff_seq[0])
				opt_results_v1['motor_eff_dmd'].append(motor_eff_seq[0])
				opt_results_v1['torque_dmd'].append(torque_seq[0])
				opt_results_v1['torque_wheel_opt'].append(-1)
				opt_results_v1['torque_wheel_dmd'].append(torque_seq[0] * gear_next * i0 * eff_diff * eff_cpling)
				opt_results_v1['torque_wheel_ctl'].append(torque_seq[0] * gear_next * i0 * eff_diff * eff_cpling)
				opt_results_v1['flag_motor_speed_ctl'].append(flag_motor_speed)
				opt_results_v1['flag_torque_ctl'].append(flag_torque)
				opt_results_v1['flag_motor_speed_dmd'].append(flag_motor_speed)
				opt_results_v1['flag_torque_dmd'].append(flag_torque)
				opt_results_v1['motor_speed_dmd'].append(motor_speed)
				opt_results_v1['motor_speed_ctl'].append(motor_speed)

				# opt_results_v1['W1'].append(0)
				# opt_results_v1['w1'].append(0)

				print(f'step {i - 15}: Invalid velocity prediction')
			else:
				
				# if vel_mor_flag == 1 :
				# 	vel_pred = np.expand_dims(np.hstack([np.linspace(vel_pred.squeeze(0)[0], vel_pred.squeeze(0)[6]*1.1, num=7),vel_pred.squeeze(0)[7:10] * 1.1]),axis=0)
				# elif vel_mor_flag == -1 :
				# 	vel_pred = np.expand_dims(np.hstack([np.linspace(vel_pred.squeeze(0)[0], vel_pred.squeeze(0)[6]*0.9, num=7),vel_pred.squeeze(0)[7:10] * 0.9]),axis=0)
				
				if i == n_steps_in:	
					# (flag, Tm_opt, vel_opt, vel_min, vel_max, vel_pred_r, gear_opt, motor_eff_opt, JcostMin, sys_mode_opt) = energy_opt_v1(gears_in_use, gear_next, vel_current=vel_current / 3.6, vel_pred=vel_pred.squeeze(0) / 3.6, vel_num_per_second=10, gearOpt=True)
					(vel_opt, gear_opt, Tm_opt, motor_eff_opt, vel_min, vel_max, flag) = energy_opt_v3(gears_in_use, gear_pre, vel_pred.squeeze(0) / 3.6, torque)
				else:
					# (flag, Tm_opt, vel_opt, vel_min, vel_max, vel_pred_r, gear_opt, motor_eff_opt, JcostMin, sys_mode_opt) = energy_opt_v1(gears_in_use, gear_ctl, vel_current=vel_current / 3.6, vel_pred=vel_pred.squeeze(0) / 3.6, vel_num_per_second=10, gearOpt=True)
					(vel_opt, gear_opt, Tm_opt, motor_eff_opt, vel_min, vel_max, flag) = energy_opt_v3(gears_in_use, gear_pre, vel_pred.squeeze(0) / 3.6, torque)
				# 转矩融合模块
				if flag == 1:
					gear_ctl = gear_opt[0]
					gear_next = gear_ctl
					_, torque_seq_ori, motor_eff_seq_dmd, flag, motor_speed_dmd = energy_and_motor_eff_calc(np.array([vel_current/3.6, vel_next/3.6]), np.array([gear_ctl]), per_meter=False)
					torque_ori = torque_seq_ori[0]

					# if abs(torque_ori - Tm_opt[0]) > 800:
					# 	torque_ctl = torque_ori
					# else:
					torque_ctl = Tm_opt[0]
					# torque_ctl = W1 * Tm_opt[0] + (1-W1) * torque_ori


					vel_ctl = vel_calc(torque_ctl, vel_current / 3.6, gear_ctl) * 3.6

					# torque_ori_list.append(torque_ori)
					# torque_ctl_list.append(torque_ctl)

					# if i == n_steps_in:
					# 	_, torque_ctl_b, _ = torque_calc(vel_current / 3.6, v[i-2]/ 3.6, gears_in_use[int(g[13] - 1)]) #14s的车速，15s的车速，14s的挡位
					# 	_, torque_ori_b, _ = torque_calc(vel_current / 3.6, v[i-2] / 3.6, gears_in_use[int(g[13] - 1)])#gears_in_use[int(g[i] - 1)]
					# 	T = torque_ctl - torque_ori
						
					# else:
					# 	T = torque_ctl_list[-1] - torque_ori_list[-1]
					# # 权重在线调节
					# w1 = w_modify(T)
					# w1_list.append(w1)
					# W1 = W1 * w1
					# if W1 > 1:
					# 	W1 = 1
					# W1_list.append(W1)

					# # 车速轨迹修正			
					# sp = (vel_pred.squeeze()[0] + vel_current) / 2 / 3.6
					# sr = (vel_current + vel_ctl ) / 2 / 3.6
					
					# Sp.append(sp)
					# Sr.append(sr) 
					# count = count + 1
					
					# if count % 10 == 0:
					# 	Sp = np.array(Sp)
					# 	Sr = np.array(Sr)
					# 	Sp = np.sum(Sp)
					# 	Sr = np.sum(Sr)
					# 	if Sp < Sr * 0.9:
					# 		vel_mor_flag = 1
					# 	elif Sp > Sr * 1.1:
					# 		vel_mor_flag = -1
					# 	else:
					# 		vel_mor_flag = 0	
					# 	Sp_sum.append(Sp)		
					# 	Sr_sum.append(Sr)
					# 	delta_S.append(Sp-Sr)
					# 	Sp = []
					# 	Sr = []
					
					vel_next = vel_ctl

					# update data_history
					data_history = np.hstack([data_history[1:], [vel_next]])

					vel_mean = distance_calc(vel_current/3.6, vel_next/3.6)
					energy, torque_seq, motor_eff_seq, flag, motor_speed = energy_and_motor_eff_calc(np.array([vel_current, vel_opt[1]])/3.6, gear_opt[:1], per_meter=False)
					opt_results_v1['flag'].append(flag)
					opt_results_v1['vel_opt'].append(vel_opt[1:])
					opt_results_v1['torque_opt'].append(Tm_opt)
					opt_results_v1['gear_opt'].append((np.where(gears_in_use == gear_ctl)[0])[0])
					opt_results_v1['motor_eff_opt'].append(motor_eff_seq[0])
					opt_results_v1['mean_vel_past'].append(vel_history.mean() * 85)
					opt_results_v1['mean_vel_pred'].append(vel_pred.mean())
					opt_results_v1['mean_vel_real'].append(vel_real.mean())
					opt_results_v1['mean_vel_opt'].append(vel_opt[1:].mean())
					opt_results_v1['vel_real'].append(vel_real)
					opt_results_v1['vel_pred'].append(vel_pred)
					opt_results_v1['vel_min'].append(vel_min)
					opt_results_v1['vel_max'].append(vel_max)
					opt_results_v1['energy_opt'].append(energy)
					opt_results_v1['Tm_wheel_diff_opt'].append((torque_ori * gear_next - Tm_opt[0] * gear_opt[0])* i0 * eff_diff * eff_cpling)
					opt_results_v1['flag_motor_speed_opt'].append(flag_motor_speed)
					opt_results_v1['flag_torque_opt'].append(flag_torque)

					opt_results_v1['vel_ctl'].append(vel_ctl)
					opt_results_v1['vel_dmd'].append(v[i])
					opt_results_v1['torque_ctl'].append(torque_ctl)
					opt_results_v1['torque_dmd'].append(torque_ori)
					opt_results_v1['gear_ctl'].append((np.where(gears_in_use == gear_ctl)[0])[0]+1)
					energy, torque_seq, motor_eff_seq, flag, motor_speed = energy_and_motor_eff_calc(np.array([vel_current, vel_ctl])/3.6, np.array([gear_ctl]), per_meter=False)
					opt_results_v1['motor_eff_ctl'].append(motor_eff_seq[0])
					opt_results_v1['energy_ctl'].append(energy)
					opt_results_v1['energy_dmd'].append(energy_dmd)
					opt_results_v1['flag_motor_speed_dmd'].append(flag_motor_speed_dmd)
					opt_results_v1['flag_torque_dmd'].append(flag_torque_dmd)
					opt_results_v1['flag_motor_speed_ctl'].append(flag_motor_speed)
					opt_results_v1['flag_torque_ctl'].append(flag_torque)
					opt_results_v1['motor_eff_dmd'].append(motor_eff_seq_dmd[0])
					opt_results_v1['motor_speed_dmd'].append(motor_speed_dmd)
					opt_results_v1['motor_speed_ctl'].append(motor_speed)


					#TODO calc Tm_wheel_diff_ctl
					torque_wheel_dmd = torque_ori * gear_next * i0 * eff_diff * eff_cpling if torque_ori > 0 else torque_ori * gear_next * i0 / (eff_diff * eff_cpling)
					torque_wheel_ctl = torque_ctl * gear_ctl * i0 * eff_diff * eff_cpling if torque_ctl > 0 else torque_ctl * gear_next * i0 / (eff_diff * eff_cpling)
					torque_wheel_opt = torque_opt * gear_opt * i0 * eff_diff * eff_cpling if torque_opt > 0 else torque_opt * gear_next * i0 / (eff_diff * eff_cpling)
					opt_results_v1['Tm_wheel_diff_ctl'].append(torque_wheel_dmd - torque_wheel_ctl)
					opt_results_v1['vel_mean'].append(vel_mean)
					
					opt_results_v1['torque_wheel_opt'].append(torque_wheel_opt)
					opt_results_v1['torque_wheel_dmd'].append(torque_wheel_dmd)
					opt_results_v1['torque_wheel_ctl'].append(torque_wheel_ctl)
					opt_results_v1['W1'].append(W1)
					opt_results_v1['w1'].append(w1)

				elif flag == 0:
					print(f'step {i - 15}: Invalid calculation')

					# update data_history
					data_history = np.hstack([data_history[1:], [vel_next]])

					gear_ctl = gear_next
					gear_next = gear_ctl
					#TODO 
					# torque_ctl = torque_ori
					vel_ctl = vel_next
					vel_mean = distance_calc(vel_current/3.6, vel_next/3.6)
					energy, torque_seq, motor_eff_seq, flag, motor_speed = energy_and_motor_eff_calc(np.array([vel_current, vel_opt[1]])/3.6, gear_opt[:1], per_meter=False)
					opt_results_v1['flag'].append(flag)
					opt_results_v1['vel_opt'].append(-1)
					opt_results_v1['torque_opt'].append(-1)
					opt_results_v1['gear_opt'].append(-1)
					opt_results_v1['motor_eff_opt'].append(-1)
					opt_results_v1['mean_vel_past'].append(vel_history.mean() * 85)
					opt_results_v1['mean_vel_pred'].append(vel_pred.mean())
					opt_results_v1['mean_vel_real'].append(vel_real.mean())
					opt_results_v1['mean_vel_opt'].append(-1)
					opt_results_v1['vel_real'].append(vel_real)
					opt_results_v1['vel_pred'].append(vel_pred)
					opt_results_v1['vel_min'].append(vel_min)
					opt_results_v1['vel_max'].append(vel_max)
					opt_results_v1['energy_opt'].append(-1)
					opt_results_v1['flag_motor_speed_opt'].append(-1)
					opt_results_v1['flag_torque_opt'].append(-1)
					
					opt_results_v1['vel_ctl'].append(vel_ctl)
					opt_results_v1['vel_dmd'].append(v[i])
					opt_results_v1['torque_ctl'].append(torque_seq[0])
					opt_results_v1['torque_dmd'].append(torque_seq[0])
					opt_results_v1['gear_ctl'].append(g[i])
					energy, torque_seq, motor_eff_seq, flag, motor_speed = energy_and_motor_eff_calc(np.array([vel_current, vel_ctl])/3.6, np.array([gear_next]), per_meter=False)
					opt_results_v1['motor_eff_ctl'].append(motor_eff_seq[0])
					opt_results_v1['motor_eff_dmd'].append(motor_eff_seq[0])
					opt_results_v1['energy_ctl'].append(energy)
					opt_results_v1['vel_mean'].append(vel_mean)
					# opt_results_v1['flag_motor_speed_ctl'].append(flag_motor_speed)
					# opt_results_v1['flag_torque_ctl'].append(flag_torque)
					# opt_results_v1['flag_motor_speed_dmd'].append(flag_motor_speed)
					# opt_results_v1['flag_torque_dmd'].append(flag_torque)
					opt_results_v1['motor_speed_dmd'].append(motor_speed)
					opt_results_v1['motor_speed_ctl'].append(motor_speed)

					opt_results_v1['torque_wheel_opt'].append(-1)
					torque_wheel_dmd = torque_seq[0] * gear_next * i0 * eff_diff * eff_cpling if torque_seq[0] > 0 else torque_seq[0] * gear_next * i0 / (eff_diff * eff_cpling)
					torque_wheel_ctl = torque_seq[0] * gear_ctl * i0 * eff_diff * eff_cpling if torque_seq[0] > 0 else torque_seq[0] * gear_next * i0 / (eff_diff * eff_cpling)

					opt_results_v1['torque_wheel_dmd'].append(torque_wheel_dmd)
					opt_results_v1['torque_wheel_ctl'].append(torque_wheel_ctl)
					
					opt_results_v1['W1'].append(0)
					opt_results_v1['w1'].append(0)

		# the past {n_steps_in} seconds of velocity exists 0
		else:
			print(f'step {i - 15}: the past {n_steps_in} seconds of velocity exists 0')

			# update data_history
			data_history = np.hstack([data_history[1:], [vel_next]])

			gear_ctl = gear_next
			gear_next = gear_ctl

			vel_mean = distance_calc(vel_current/3.6, vel_next/3.6)
			energy, torque_seq, motor_eff_seq, flag, motor_speed = energy_and_motor_eff_calc(np.array([vel_current, vel_next])/3.6, np.array([gear_next]), per_meter=False)
			opt_results_v1['invalid_indexes'].append(i)
			opt_results_v1['flag'].append(2)
			opt_results_v1['vel_opt'].append(-1)
			opt_results_v1['gear_opt'].append(-1)
			opt_results_v1['torque_opt'].append(-1)
			opt_results_v1['motor_eff_opt'].append(-1)
			opt_results_v1['mean_vel_past'].append(-1)
			opt_results_v1['mean_vel_pred'].append(-1)
			opt_results_v1['mean_vel_real'].append(-1)	
			opt_results_v1['mean_vel_opt'].append(-1)
			opt_results_v1['vel_real'].append(-1)
			opt_results_v1['vel_pred'].append(-1)
			opt_results_v1['vel_min'].append(-1)
			opt_results_v1['vel_max'].append(-1)
			opt_results_v1['energy_opt'].append(-1)
			opt_results_v1['if_opt'].append(0)
			opt_results_v1['vel_mean'].append(vel_mean)
			opt_results_v1['flag_motor_speed_opt'].append(-1)
			opt_results_v1['flag_torque_opt'].append(-1)

			opt_results_v1['vel_ctl'].append(vel_next)
			opt_results_v1['vel_dmd'].append(v[i])
			energy, torque_seq, motor_eff_seq, flag, motor_speed = energy_and_motor_eff_calc(np.array([vel_current, vel_next])/3.6, np.array([gear_next]), per_meter=False)
			opt_results_v1['torque_ctl'].append(torque_seq[0])
			opt_results_v1['torque_dmd'].append(torque_seq[0])
			opt_results_v1['gear_ctl'].append(g[i])			 
			opt_results_v1['motor_eff_ctl'].append(motor_eff_seq[0])
			opt_results_v1['motor_eff_dmd'].append(motor_eff_seq[0])
			opt_results_v1['energy_ctl'].append(energy)
			# opt_results_v1['flag_motor_speed_ctl'].append(flag_motor_speed)
			# opt_results_v1['flag_motor_speed_dmd'].append(flag_motor_speed)
			# opt_results_v1['flag_torque_ctl'].append(flag_torque)
			# opt_results_v1['flag_torque_dmd'].append(flag_torque)
			opt_results_v1['motor_speed_dmd'].append(motor_speed)
			opt_results_v1['motor_speed_ctl'].append(motor_speed)


			opt_results_v1['torque_wheel_opt'].append(-1)
			torque_wheel_dmd = torque_seq[0] * gear_next * i0 * eff_diff * eff_cpling if torque_ori > 0 else torque_ori * gear_next * i0 / (eff_diff * eff_cpling)
			torque_wheel_ctl = torque_seq[0] * gear_ctl * i0 * eff_diff * eff_cpling if torque_ctl > 0 else torque_ctl * gear_next * i0 / (eff_diff * eff_cpling)

			opt_results_v1['torque_wheel_dmd'].append(torque_wheel_dmd)
			opt_results_v1['torque_wheel_ctl'].append(torque_wheel_ctl)

			opt_results_v1['W1'].append(0)
			opt_results_v1['w1'].append(0)

	else:
		print(f'step {i - 15}: does not enter optimization module\n')
		# update data_history
		data_history = np.hstack([data_history[1:], [vel_next]])

		gear_ctl = gear_next
		gear_next = gear_ctl

		vel_mean = distance_calc(vel_current/3.6, vel_next/3.6)
		energy, torque_seq, motor_eff_seq, flag, motor_speed = energy_and_motor_eff_calc(np.array([vel_current, vel_next])/3.6, np.array([gear_next]), per_meter=False)
		opt_results_v1['flag'].append(3)
		opt_results_v1['flag_motor_speed_opt'].append(-1)
		opt_results_v1['flag_torque_opt'].append(-1)
		opt_results_v1['vel_opt'].append(-1)
		opt_results_v1['gear_opt'].append(-1)
		opt_results_v1['motor_eff_opt'].append(-1)
		opt_results_v1['mean_vel_past'].append(-1)
		opt_results_v1['mean_vel_pred'].append(-1)
		opt_results_v1['mean_vel_real'].append(-1)
		opt_results_v1['mean_vel_opt'].append(-1)
		opt_results_v1['vel_real'].append(-1)
		opt_results_v1['vel_pred'].append(-1)
		opt_results_v1['vel_min'].append(-1)
		opt_results_v1['vel_max'].append(-1)
		opt_results_v1['energy_opt'].append(-1)
		opt_results_v1['torque_opt'].append(-1)
		opt_results_v1['if_opt'].append(0)
		opt_results_v1['vel_mean'].append(vel_mean)
		opt_results_v1['torque_wheel_opt'].append(-1)

		torque_wheel_dmd = torque_seq[0] * gear_next * i0 * eff_diff * eff_cpling if torque_seq[0] > 0 else torque_seq[0] * gear_next * i0 / (eff_diff * eff_cpling)
		opt_results_v1['torque_wheel_dmd'].append(torque_wheel_dmd)

		opt_results_v1['W1'].append(0)
		opt_results_v1['w1'].append(0)


		opt_results_v1['vel_ctl'].append(v[i])
		opt_results_v1['vel_dmd'].append(v[i])
		opt_results_v1['gear_ctl'].append(g[i])			 
		energy, torque_seq, motor_eff_seq, flag, motor_speed = energy_and_motor_eff_calc(np.array([vel_current, vel_next])/3.6, np.array([gear_next]), per_meter=False)
		opt_results_v1['motor_eff_ctl'].append(motor_eff_seq[0])
		opt_results_v1['motor_eff_dmd'].append(motor_eff_seq[0])
		# opt_results_v1['flag_motor_speed_ctl'].append(flag_motor_speed)	
		# opt_results_v1['flag_motor_speed_dmd'].append(flag_motor_speed)
		# opt_results_v1['flag_torque_ctl'].append(flag_torque)
		# opt_results_v1['flag_torque_dmd'].append(flag_torque)		
		opt_results_v1['torque_ctl'].append(torque_seq[0])
		opt_results_v1['torque_dmd'].append(torque_seq[0])
		opt_results_v1['energy_ctl'].append(energy)
		torque_wheel_ctl = torque_wheel_dmd
		opt_results_v1['torque_wheel_ctl'].append(torque_wheel_ctl)
		opt_results_v1['motor_speed_dmd'].append(motor_speed)
		opt_results_v1['motor_speed_ctl'].append(motor_speed)
		

	# print(opt_results_v1['flag'][count])
	
opt_results_v1['energy_total_opt'] = sum(opt_results_v1['energy_opt']) / 3600 / 1000  # kw*h
opt_results_v1['single_vel_opt'] = [x[0] if isinstance(x, np.ndarray) else x for x in opt_results_v1['vel_opt']]
opt_results_v1['single_torque_opt'] = [x[0] if isinstance(x, np.ndarray) else x for x in opt_results_v1['torque_opt']]
opt_results_v1['single_gear_opt'] = [x[0] if isinstance(x, np.ndarray) else x for x in opt_results_v1['gear_opt']]
opt_results_v1['single_motor_eff_opt'] = [x[0] if isinstance(x, np.ndarray) else x for x in opt_results_v1['motor_eff_opt']]

opt_results_v1['energy_total_ctl'] = sum(opt_results_v1['energy_ctl']) / 3600 / 1000  # kw*h
opt_results_v1['energy_total_dmd'] = sum(opt_results_v1['energy_dmd']) / 3600 / 1000  # kw*h
opt_results_v1['distance'] = sum(opt_results_v1['vel_mean'])
    
	# results_1_noW[string] = opt_results_v1
#%% calculate original total energy

#%%
# for idx in range(len(sorted_indexes)):
# 	if sorted_indexes[idx].size > 10 * 60 * 5:  # 10 mins
# results_ori_noW = {}
# for q in [0,1,2,3]:
# 	q = str(q)
# 	string = 'v_idx' + q
# 	results_ori_noW[string] = {}
# 	q = int(q)

	# idx = 0  # the longest
	# V = velocity[sorted_indexes[idx]]
	# A = acc[sorted_indexes[idx]]
	# B = brake[sorted_indexes[idx]]
	# G = gear[sorted_indexes[idx]]

	# v = np.array([V[i:i + vel_count_per_second].mean() for i in range(0, V.size, vel_count_per_second) if i + vel_count_per_second <= V.size])

	# # if an average is used, there may be cases where both the accelerator and brake are greater than 0
	# a = np.array([A[i] for i in range(0, A.size, vel_count_per_second) if i + vel_count_per_second <= A.size])
	# b = np.array([B[i] for i in range(0, B.size, vel_count_per_second) if i + vel_count_per_second <= B.size])
	# g = np.array([G[i] for i in range(0, G.size, vel_count_per_second) if i + vel_count_per_second <= G.size])

	# data = np.vstack((v, a)) 
	# data = moving_average(data)  # (n_features, )
	# v = moving_average(v)

	# mode = 'min-max'
	# assert data[0].shape == g.shape == b.shape == a.shape



ori_results = {}
ori_results['vel'] = []
ori_results['gear'] = []
ori_results['torque'] = []			
ori_results['energy'] = []
ori_results['motor_eff'] = []
ori_results['energy_total'] = 0
ori_results['vel_mean'] = []
ori_results['torque_wheel'] = []

for i in progressbar.progressbar(range(n_steps_in, v.size - n_steps_out)):
# for i in progressbar.progressbar(range(n_steps_in, n_steps_in+20)):
	vel_current = v[i - 1] / 3.6
	vel_next = v[i] / 3.6
	gear_next = gears_in_use[int(g[i] - 1)]
	(energy, motor_seq, motor_eff_seq, _,_) = energy_and_motor_eff_calc(v[i-1:i+1]/3.6, np.array([gear_next]), per_meter=False)
	vel_mean = (vel_next + vel_current) / 2
	ori_results['vel'].append(v[i])
	ori_results['gear'].append(g[i])
	ori_results['torque'].append(motor_seq[0])
	ori_results['energy'].append(energy)

	torque_wheel = motor_seq[0] * gear_next * i0 * eff_diff * eff_cpling if motor_seq[0] > 0 else motor_seq[0] * gear_next * i0 / (eff_diff * eff_cpling) 
	ori_results['torque_wheel'].append(torque_wheel)
	ori_results['motor_eff'].append(motor_eff_seq[0])
	ori_results['vel_mean'].append(vel_mean)
ori_results['energy_total'] = sum(ori_results['energy']) / 3600 / 1000

#%% distance
ori_results['distance'] = sum(ori_results['vel_mean'])
# results_ori_noW[string] = ori_results

# %%
# with open('opt_results_v1.pickle', 'wb') as f:
#  	pickle.dump([opt_results_v1, ori_results],f)
	 
# with open('opt_results_v1.pickle','rb') as f:
# 	[opt_results_v1,ori_results] = pickle.load(f)