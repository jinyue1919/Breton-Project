#%% Imports
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.signal
from sklearn import preprocessing

def acc_intention(acc_pedal, acc_drt):
	# New Antecedent/Consequent objects hold universe variables and membership
	acc = ctrl.Antecedent(np.linspace(0, 1, 21), 'acceleration')
	acc_derivative = ctrl.Antecedent(np.linspace(-1, 1, 81), 'derivative of acceleration')
	driver_intention = ctrl.Consequent(np.linspace(0, 1, 21), 'driver intention')

	# Membership functions
	acc['S'] = fuzz.trapmf(acc.universe, [0, 0, 0.15, 0.25])
	acc['RS'] = fuzz.trimf(acc.universe, [0.1, 0.25, 0.4])
	acc['M'] = fuzz.trimf(acc.universe, [0.3, 0.45, 0.6])
	acc['RB'] = fuzz.trimf(acc.universe, [0.5, 0.65, 0.8])
	acc['B'] = fuzz.trapmf(acc.universe, [0.7, 0.85, 1, 1])

	acc_derivative['NB'] = fuzz.trapmf(acc_derivative.universe, [-1, -1, -0.7, -0.4])
	acc_derivative['NS'] = fuzz.trimf(acc_derivative.universe, [-0.6, -0.225, 0.15])
	acc_derivative['S'] = fuzz.trimf(acc_derivative.universe, [-0.15, 0.1, 0.35])
	acc_derivative['M'] = fuzz.trimf(acc_derivative.universe, [0.3, 0.45, 0.6])
	acc_derivative['B'] = fuzz.trapmf(acc_derivative.universe, [0.5, 0.7, 1, 1])

	driver_intention['NL'] = fuzz.trapmf(driver_intention.universe, [0, 0, 0.1, 0.15])
	driver_intention['NS'] = fuzz.trimf(driver_intention.universe, [0.1, 0.25, 0.4])
	driver_intention['ZE'] = fuzz.trimf(driver_intention.universe, [0.3, 0.45, 0.6])
	driver_intention['PS'] = fuzz.trimf(driver_intention.universe, [0.5, 0.65, 0.8])
	driver_intention['PL'] = fuzz.trapmf(driver_intention.universe, [0.7, 0.8, 1, 1])

	acc.view()
	acc_derivative.view()
	driver_intention.view()

	# Define rule base
	rule1 = ctrl.Rule(acc['S'] & acc_derivative['NB'], driver_intention['NL'])
	rule2 = ctrl.Rule(acc['S'] & acc_derivative['NS'], driver_intention['NL'])
	rule3 = ctrl.Rule(acc['S'] & acc_derivative['S'], driver_intention['NS'])
	rule4 = ctrl.Rule(acc['S'] & acc_derivative['M'], driver_intention['NS'])
	rule5 = ctrl.Rule(acc['S'] & acc_derivative['B'], driver_intention['ZE'])

	rule6 = ctrl.Rule(acc['RS'] & acc_derivative['NB'], driver_intention['NL'])
	rule7 = ctrl.Rule(acc['RS'] & acc_derivative['NS'], driver_intention['NS'])
	rule8 = ctrl.Rule(acc['RS'] & acc_derivative['S'], driver_intention['NS'])
	rule9 = ctrl.Rule(acc['RS'] & acc_derivative['M'], driver_intention['ZE'])
	rule10 = ctrl.Rule(acc['RS'] & acc_derivative['B'], driver_intention['PS'])

	rule11 = ctrl.Rule(acc['M'] & acc_derivative['NB'], driver_intention['NS'])
	rule12 = ctrl.Rule(acc['M'] & acc_derivative['NS'], driver_intention['ZE'])
	rule13 = ctrl.Rule(acc['M'] & acc_derivative['S'], driver_intention['ZE'])
	rule14 = ctrl.Rule(acc['M'] & acc_derivative['M'], driver_intention['PS'])
	rule15 = ctrl.Rule(acc['M'] & acc_derivative['B'], driver_intention['PL'])

	rule16 = ctrl.Rule(acc['RB'] & acc_derivative['NB'], driver_intention['NS'])
	rule17 = ctrl.Rule(acc['RB'] & acc_derivative['NS'], driver_intention['ZE'])
	rule18 = ctrl.Rule(acc['RB'] & acc_derivative['S'], driver_intention['PS'])
	rule19 = ctrl.Rule(acc['RB'] & acc_derivative['M'], driver_intention['PS'])
	rule20 = ctrl.Rule(acc['RB'] & acc_derivative['B'], driver_intention['PL'])

	rule21 = ctrl.Rule(acc['B'] & acc_derivative['NB'], driver_intention['ZE'])
	rule22 = ctrl.Rule(acc['B'] & acc_derivative['NS'], driver_intention['ZE'])
	rule23 = ctrl.Rule(acc['B'] & acc_derivative['S'], driver_intention['PS'])
	rule24 = ctrl.Rule(acc['B'] & acc_derivative['M'], driver_intention['PL'])
	rule25 = ctrl.Rule(acc['B'] & acc_derivative['B'], driver_intention['PL'])

	# rule1.view()


	# Create control systems
	driver_intention_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9,
	rule10, rule11, rule12, rule12, rule13, rule14, rule15, rule16, rule16, rule17, rule18, rule19, rule20,
	rule21, rule22, rule23, rule24, rule25])

	driver_intention_fc = ctrl.ControlSystemSimulation(driver_intention_ctrl)

	driver_intention_fc.input['acceleration'] = acc_pedal
	driver_intention_fc.input['derivative of acceleration'] = acc_drt

	driver_intention_fc.compute()
	
	# print(driver_intention_fc.output['driver intention'])
	# driver_intention.view(sim=driver_intention_fc)
	# return driver_intention_fc.output['driver intention']

def brake_intention(brake_pedal, brake_drt):
	brake = ctrl.Antecedent(np.linspace(0, 1, 81), 'brake')
	brake_derivative = ctrl.Antecedent(np.linspace(-1, 1, 5121), 'derivative of brake')
	driver_intention = ctrl.Consequent(np.linspace(-1, 0, 81), 'driver intention')
	
	# Membership functions
	# TODO: search papers to check membership functions
	brake['S'] = fuzz.trapmf(brake.universe, [0, 0, 0.05, 0.1])
	brake['RS'] = fuzz.trimf(brake.universe, [0, 0.075, 0.15])
	brake['M'] = fuzz.trimf(brake.universe, [0.1, 0.15, 0.2])
	brake['RB'] = fuzz.trimf(brake.universe, [0.15, 0.225, 0.3])
	brake['B'] = fuzz.trapmf(brake.universe, [0.2, 0.3, 1, 1])
	
	brake_derivative['NB'] = fuzz.trapmf(brake_derivative.universe, [-1, -1, -0.14, -0.08])
	brake_derivative['NS'] = fuzz.trimf(brake_derivative.universe, [-0.12, -0.05, 0.02])
	brake_derivative['S'] = fuzz.trimf(brake_derivative.universe, [-0.02, 0.02, 0.06])
	brake_derivative['M'] = fuzz.trimf(brake_derivative.universe, [0.06, 0.09, 0.12])
	brake_derivative['B'] = fuzz.trapmf(brake_derivative.universe, [0.1, 0.14, 1, 1])
	
	driver_intention['NL'] = fuzz.trapmf(driver_intention.universe, [-1, -1, -0.9, -0.85])
	driver_intention['NS'] = fuzz.trimf(driver_intention.universe, [-0.9, -0.75, -0.6])
	driver_intention['ZE'] = fuzz.trimf(driver_intention.universe, [-0.7, -0.55, -0.4])
	driver_intention['PS'] = fuzz.trimf(driver_intention.universe, [-0.5, -0.35, -0.2])
	driver_intention['PL'] = fuzz.trapmf(driver_intention.universe, [-0.3, -0.2, 0, 0])
	
	# brake.view()
	# brake_derivative.view()
	# driver_intention.view()
	
	#  Define rule base
	rule1 = ctrl.Rule(brake['S'] & brake_derivative['NB'], driver_intention['PL'])
	rule2 = ctrl.Rule(brake['S'] & brake_derivative['NS'], driver_intention['PL'])
	rule3 = ctrl.Rule(brake['S'] & brake_derivative['S'], driver_intention['PS'])
	rule4 = ctrl.Rule(brake['S'] & brake_derivative['M'], driver_intention['PS'])
	rule5 = ctrl.Rule(brake['S'] & brake_derivative['B'], driver_intention['ZE'])
	
	rule6 = ctrl.Rule(brake['RS'] & brake_derivative['NB'], driver_intention['PL'])
	rule7 = ctrl.Rule(brake['RS'] & brake_derivative['NS'], driver_intention['PS'])
	rule8 = ctrl.Rule(brake['RS'] & brake_derivative['S'], driver_intention['PS'])
	rule9 = ctrl.Rule(brake['RS'] & brake_derivative['M'], driver_intention['ZE'])
	rule10 = ctrl.Rule(brake['RS'] & brake_derivative['B'], driver_intention['ZE'])
	
	rule11 = ctrl.Rule(brake['M'] & brake_derivative['NB'], driver_intention['PS'])
	rule12 = ctrl.Rule(brake['M'] & brake_derivative['NS'], driver_intention['PS'])
	rule13 = ctrl.Rule(brake['M'] & brake_derivative['S'], driver_intention['ZE'])
	rule14 = ctrl.Rule(brake['M'] & brake_derivative['M'], driver_intention['ZE'])
	rule15 = ctrl.Rule(brake['M'] & brake_derivative['B'], driver_intention['NS'])
	
	rule16 = ctrl.Rule(brake['RB'] & brake_derivative['NS'], driver_intention['PS'])
	rule17 = ctrl.Rule(brake['RB'] & brake_derivative['S'], driver_intention['ZE'])
	rule18 = ctrl.Rule(brake['RB'] & brake_derivative['S'], driver_intention['ZE'])
	rule19 = ctrl.Rule(brake['RB'] & brake_derivative['M'], driver_intention['NS'])
	rule20 = ctrl.Rule(brake['RB'] & brake_derivative['B'], driver_intention['NL'])
	
	rule21 = ctrl.Rule(brake['B'] & brake_derivative['NB'], driver_intention['ZE'])
	rule22 = ctrl.Rule(brake['B'] & brake_derivative['NS'], driver_intention['ZE'])
	rule23 = ctrl.Rule(brake['B'] & brake_derivative['S'], driver_intention['NS'])
	rule24 = ctrl.Rule(brake['B'] & brake_derivative['M'], driver_intention['NL'])
	rule25 = ctrl.Rule(brake['B'] & brake_derivative['B'], driver_intention['NL'])
	
	# rule1.view()
	
	# Create control systems
	driver_intention_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9,
	rule10, rule11, rule12, rule12, rule13, rule14, rule15, rule16, rule16, rule17, rule18, rule19, rule20,
	rule21, rule22, rule23, rule24, rule25])
	
	driver_intention_fc = ctrl.ControlSystemSimulation(driver_intention_ctrl)
	
	driver_intention_fc.input['brake'] = brake_pedal
	driver_intention_fc.input['derivative of brake'] = brake_drt
	
	driver_intention_fc.compute()
	
	# print(driver_intention_fc.output['driver intention'])
	# driver_intention.view(sim=driver_intention_fc)
	return driver_intention_fc.output['driver intention']

#%% test
if ('__main__' == __name__):
	files = os.listdir('BretonDataFive')
	files.sort()
	fileName = os.path.join('BretonDataFive', files[0])
	
	df = pd.read_csv(fileName,
	    header=None, sep='\s+', names=['vel', 'acc', 'brake', 'gear', 'gearFlag'],
	    dtype={'vel': np.float32, 'acc': np.float32, 'brake': np.float32, 'gear': np.float32, 'gearFlag': np.int})
	velocity = df['vel'].to_numpy()
	acc = df['acc'].to_numpy()
	brake = df['brake'].to_numpy()
	gear = df['gear'].to_numpy()
	gearFlag = df['gearFlag'].to_numpy()
	
	vel_count_per_second = 5
	vel_seq_temp_list = [velocity[i:i + vel_count_per_second].mean() for i in range(0, velocity.size, vel_count_per_second) if (i + vel_count_per_second <= velocity.size)]
	vel_seq = scipy.signal.medfilt(vel_seq_temp_list, kernel_size=15)
	acc_seq = np.array([acc[i + vel_count_per_second - 1] for i in range(0, acc.size, vel_count_per_second) if (i + vel_count_per_second <= acc.size)])
	brake_seq = np.array([brake[i + vel_count_per_second - 1] for i in range(0, brake.size, vel_count_per_second) if (i + vel_count_per_second <= brake.size)])
	gear_seq = np.array([gear[i + vel_count_per_second - 1] for i in range(0, gear.size, vel_count_per_second) if (i + vel_count_per_second <= gear.size)])
	gearFlag_seq = np.array([gearFlag[i + vel_count_per_second - 1] for i in range(0, gearFlag.size, vel_count_per_second) if (i + vel_count_per_second <= gearFlag.size)])
	
	# Check data correctness
	# print(vel_seq.size, acc_seq.size, brake_seq.size, gear_seq.size, gearFlag_seq.size)
	
	#%% figure
	plt.plot(vel_seq, label='velocity')
	plt.xlabel('time (s)')
	plt.ylabel('velocity: km/h')
	plt.legend(loc='upper right')
	plt.show()
	
	plt.plot(acc_seq * 100, label='acc pedal')
	plt.plot(brake_seq * 100, label='brake pedal')
	plt.xlabel('time (s)')
	plt.ylabel('%')
	plt.legend(loc='upper right')
	plt.show()
	
	
	plt.plot(gear_seq, label='gear')
	plt.plot(gearFlag_seq, label='gear flag')
	plt.xlabel('time (s)')
	plt.text(2, 4, 'gear flag\n-1: R\n0 and 2: N\n1: D\n3: R')
	plt.legend(loc='upper right')
	plt.show()
	
	#%% validation
	driver_intention = []
	driver_intention.append(0)
	vel_delta = []
	for i in range(1, acc_seq.size):
		vel_delta.append(vel_seq[i] - vel_seq[i - 1])
	
		if (acc_seq[i - 1] > 0) and (acc_seq[i] > 0):
			acc_delta = acc_seq[i] - acc_seq[i - 1]
			driver_intention.append(acc_intention(acc_seq[i], acc_delta))
		elif (brake_seq[i - 1] > 0) and (brake_seq[i] > 0):
			brake_delta = brake_seq[i] - brake_seq[i - 1]
			driver_intention.append(brake_intention(brake_seq[i], brake_delta))
		else:
			driver_intention.append(driver_intention[i - 1])
	
	vel_delta = np.array(vel_delta)
	vel_delta = 2 * (vel_delta - np.amin(vel_delta)) / (np.amax(vel_delta) - np.amin(vel_delta)) - 1
	plt.plot(driver_intention, label='driver intention')
	plt.plot(vel_delta, label='normalized accleration')
	plt.legend()
	plt.xlabel('time (s)')
	plt.show()
	
	
	#%%
	acc_drt = []
	brake_drt = []
	for i in range(1, acc_seq.size):
		acc_drt.append(acc_seq[i] - acc_seq[i - 1])
		brake_drt.append(brake_seq[i] - brake_seq[i - 1])
	plt.plot(acc_drt)
	plt.plot(brake_drt)
	plt.show()
	
	
	
	
	
	
