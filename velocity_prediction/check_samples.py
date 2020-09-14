import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True, linewidth=200)

Path2VelFile = os.path.join(os.getcwd(), 'Data')
VelFiles = os.listdir(Path2VelFile)
VelFiles.sort()
abnormal_vel_idx = {}

for count, file in enumerate(VelFiles):
	PathFinal = os.path.join(Path2VelFile, file)
	vel = pd.read_csv(PathFinal, header=None, names=['velocity']).to_numpy()
	if (vel[np.nonzero(vel > 85)].size > 0):
		abnormal_vel_idx[file] = (np.nonzero(vel > 85))
		print('Found abnormal velocity in file:', file, ', at line:', np.nonzero(vel > 85)[0]),

abnormal_vel_files = [*abnormal_vel_idx]  # extract all

# Plot abnormal velocity 
for file in abnormal_vel_files:
	PathFinal = os.path.join(Path2VelFile, file)
	vel = pd.read_csv(PathFinal, header=None, names=['velocity']).to_numpy()
	plt.plot(vel)
	plt.ylabel(file)
	plt.show()
