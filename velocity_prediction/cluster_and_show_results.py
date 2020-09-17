# import os
from vt_optimize import *
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

with open('res.pickle','rb') as f:
    [res_rec] = pickle.load(f)     # v_test : m/s  distance : m

def distance_cal(v):
    v_mean = np.array([v[i:i + 2].mean() for i in range(len(v) - 1)])             
    distance = np.sum(v_mean)
    return distance

# 把优化的结果和原始的结果 （能耗、速度轨迹、里程）整理在一个dictionary里
result_compar = {}
ori = {}
opt = {}
ori['distance'] = []
ori['energy'] = []
ori['v_t'] = []
opt['distance'] = []
opt['energy'] = []
opt['v_t'] = []
# calculate opt 
for i in range(len(res_rec)):
    distance_opt = distance_cal(res_rec[i].x)
    opt['distance'].append(distance_opt)
    opt['energy'].append(res_rec[i].fun)
    opt['v_t'].append(res_rec[i].x)

# calculate ori 
for i in range(len(res_rec)):
    energy_ori = energy(v_test[i])
    ori['energy'].append(energy_ori)
    ori['distance'].append(distance[i])
    ori['v_t'].append(v_test[i])

result_compar['opt'] = opt
result_compar['ori'] = ori

with open('result_compar.pickle', 'wb') as f:
    pickle.dump([result_compar], f)

#%% find rules
# calculate v_average and v_standard deviation of ori 
v_aver = [result_compar['ori']['v_t'][i].mean() for i in range(len(result_compar['ori']['v_t']))]
v_standard = [result_compar['ori']['v_t'][i].std() for i in range(len(result_compar['ori']['v_t']))]

# clustering
X = np.array([[v_aver[i], v_standard[i] ]for i in range(len(v_standard))])
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
label = kmeans.labels_

# 按label将优化结果和原始结果存储为字典
idx0 = list(np.where(label == 0)[0]) 
idx1 = list(np.where(label == 1)[0]) 
idx2 = list(np.where(label == 2)[0]) 
idx3 = list(np.where(label == 3)[0]) 
idx4 = list(np.where(label == 4)[0]) 

result_cluster = {}

cluster0 = {}
cluster1 = {}
cluster2 = {}
cluster3 = {}
cluster4 = {}

opt0 = {}
opt1 = {}
opt2 = {}
opt3 = {}
opt4 = {}

ori0 = {}
ori1 = {}
ori2 = {}
ori3 = {}
ori4 = {}


# 速度轨迹
opt0['v_t'] = np.array(result_compar['opt']['v_t'])[idx0]
opt1['v_t'] = np.array(result_compar['opt']['v_t'])[idx1]
opt2['v_t'] = np.array(result_compar['opt']['v_t'])[idx2]
opt3['v_t'] = np.array(result_compar['opt']['v_t'])[idx3]
opt4['v_t'] = np.array(result_compar['opt']['v_t'])[idx4]

ori0['v_t'] = np.array(result_compar['ori']['v_t'])[idx0]
ori1['v_t'] = np.array(result_compar['ori']['v_t'])[idx1]
ori2['v_t'] = np.array(result_compar['ori']['v_t'])[idx2]
ori3['v_t'] = np.array(result_compar['ori']['v_t'])[idx3]
ori4['v_t'] = np.array(result_compar['ori']['v_t'])[idx4]

# 能耗
opt0['energy'] = np.array(result_compar['opt']['energy'])[idx0]
opt1['energy'] = np.array(result_compar['opt']['energy'])[idx1]
opt2['energy'] = np.array(result_compar['opt']['energy'])[idx2]
opt3['energy'] = np.array(result_compar['opt']['energy'])[idx3]
opt4['energy'] = np.array(result_compar['opt']['energy'])[idx4]

ori0['energy'] = np.array(result_compar['ori']['energy'])[idx0]
ori1['energy'] = np.array(result_compar['ori']['energy'])[idx1]
ori2['energy'] = np.array(result_compar['ori']['energy'])[idx2]
ori3['energy'] = np.array(result_compar['ori']['energy'])[idx3]
ori4['energy'] = np.array(result_compar['ori']['energy'])[idx4]

# 距离
opt0['distance'] = np.array(result_compar['opt']['distance'])[idx0]
opt1['distance'] = np.array(result_compar['opt']['distance'])[idx1]
opt2['distance'] = np.array(result_compar['opt']['distance'])[idx2]
opt3['distance'] = np.array(result_compar['opt']['distance'])[idx3]
opt4['distance'] = np.array(result_compar['opt']['distance'])[idx4]
ori0['distance'] = np.array(result_compar['ori']['distance'])[idx0]
ori1['distance'] = np.array(result_compar['ori']['distance'])[idx1]
ori2['distance'] = np.array(result_compar['ori']['distance'])[idx2]
ori3['distance'] = np.array(result_compar['ori']['distance'])[idx3]
ori4['distance'] = np.array(result_compar['ori']['distance'])[idx4]

cluster0['opt0'] = opt0
cluster0['ori0'] = ori0

cluster1['opt1'] = opt1
cluster1['ori1'] = ori1

cluster2['opt2'] = opt2
cluster2['ori2'] = ori2

cluster3['opt3'] = opt3
cluster3['ori3'] = ori3

cluster4['opt4'] = opt4
cluster4['ori4'] = ori4

result_cluster['cluster0'] = cluster0
result_cluster['cluster1'] = cluster1
result_cluster['cluster2'] = cluster2
result_cluster['cluster3'] = cluster3
result_cluster['cluster4'] = cluster4

with open('result_cluster.pickle', 'wb') as f:
 	pickle.dump([result_cluster], f)


# 画图
for i in range(len(result_cluster['cluster4']['ori4']['v_t'])):
    plt.plot(result_cluster['cluster4']['ori4']['v_t'][i], label = 'ori')
    plt.plot(result_cluster['cluster4']['opt4']['v_t'][i], label = 'opt')
    plt.legend()
    plt.show()

# result_cluster['cluster0']['opt0']['distance'][0] / result_cluster['cluster0']['ori0']['distance'][0] - 1
# result_cluster['cluster0']['opt0']['energy'][0] / result_cluster['cluster0']['ori0']['energy'][0] - 1

# print(f'distance optimize: {result_cluster['cluster0']['opt0']['distance'][0] / result_cluster['cluster0']['opt0']['distance'][0] - 1}') 
# print(f'energy optimize：{result_cluster['cluster0']['opt0']['energy'][0] / result_cluster['cluster0']['opt0']['energy'][0] - 1}')
# # # %%
# print(f'\nmulti-step: original motor speed exceed limits, set to largest.\n'
# 					f'delta motor speed: {delta_speed}\n'
# 					f'acc: {vel_seq[1] - vel_seq[0]} m/s\n'
# 					f'original motor_speed: {motor_speed}\n'
# 					f'gear: {gear_seq[0]}\n')