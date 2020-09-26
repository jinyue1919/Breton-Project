import os
import time
import progressbar
from util_vel_pred import *
from energy_opt_v1 import *
from energy_opt_v2 import *
import pickle
from fuzzy_w_modify import *
import math
import pdb
from scipy.optimize import minimize
import pickle 

def extract_nonzero(data):
    idx_nonzero = np.where(data != 0)[0]
    idx_diff = np.diff(idx_nonzero)
    idx_consecutive = np.where(idx_diff != 1)[0]
    idx = np.split(idx_nonzero, idx_consecutive + 1)
    return idx

#%% load and preprocess data   
vel_CAN_cycle = 0.2  # unit: s
vel_count_per_second = int(1 / vel_CAN_cycle)

dataFolder = 'BretonDataTest'
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

V = velocity[sorted_indexes[0]]

v = np.array([V[i:i + vel_count_per_second].mean() for i in range(0, V.size, vel_count_per_second) if i + vel_count_per_second <= V.size])

#TODO
idx = extract_nonzero(v)
idx = sorted(idx, key=lambda x: x.size, reverse=True)
velocity = v[idx[0]] / 3.6  # velocity : m/s
v_km = v[idx[0]]            # v_km :km/h
# calculate distance in 10s
t_window = 11
# #calculate distance in 5s
# t_window = 6
v_mean = np.array([velocity[i:i + 2].mean() for i in range(len(velocity) - 1)])             # v_mean : m/s
distance = [velocity[i:i + t_window].sum() for i in range(len(velocity) - t_window + 1)]    # distance : m
v_test = np.array([velocity[i:i+t_window]  for i in range(len(velocity) - t_window + 1)])   #  v_test : m/s 

# distance_5s = [velocity[i:i + t_window].sum() for i in range(len(velocity) - t_window + 1)]    # distance : m
# v_test_5s = np.array([velocity[i:i+t_window]  for i in range(len(velocity) - t_window + 1)])   #  v_test : m/s 

#%% save data
with open('preprocessing_data.pickle', 'wb') as f:
 	pickle.dump([v_test, distance, v_km], f)

# with open('preprocessing_data_5s.pickle', 'wb') as f:
#  	pickle.dump([v_test_5s, distance_5s, v_km], f)