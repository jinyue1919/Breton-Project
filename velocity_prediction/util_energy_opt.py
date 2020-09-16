#%% imports
# import os
# os.chdir('D:\jinyue1919\Documents\研究生\磕盐\搬砖\纯电动\资料')
import numpy as np
import pandas as pd
import math
from scipy import interpolate

np.set_printoptions(precision=4, suppress=True, linewidth=200)

#%% Motor efficiency map (the same with or without gear optimization) (Efficiency when motor = 0 and speed = 0 has been manually added)
motor_pos_speeds = np.array([speed for speed in range(0, 3300, 300)]) * (2 * np.pi / 60)  # unit: rad/s; manually add speed = 0
motor_neg_speeds = np.array([0, 300, 600, 900, 1200, 1600, 1900, 2200, 2500, 2800, 3000]) * (2 * np.pi / 60)
motor_pos_torque = np.arange(0, 2700, 100)  # Manually add torque = 0
motor_neg_torque = -motor_pos_torque[::-1]

motor_eff_pos_map = pd.read_csv('motor_eff_pos_map.csv', header=None, names=motor_pos_speeds).values
motor_eff_neg_map = pd.read_csv('motor_eff_neg_map.csv', header=None, names=motor_neg_speeds).values[::-1]

# External characteristics
motor_max_pos_torque = np.array([2570, 2570, 2543, 2532, 2532, 2217, 1844, 1580, 1381, 1229, 1119])  
motor_max_neg_torque = -1 * np.array([2596, 2596, 2586, 2594, 2590, 2140, 1783, 1531, 1340, 1203, 1084])

interpolate_pos_torque = interpolate.interp1d(motor_pos_speeds, motor_max_pos_torque)
interpolate_neg_torque = interpolate.interp1d(motor_neg_speeds, motor_max_neg_torque)
interpolate_pos_motor_eff = interpolate.interp2d(motor_pos_speeds, motor_pos_torque, motor_eff_pos_map)
interpolate_neg_motor_eff = interpolate.interp2d(motor_neg_speeds, motor_neg_torque, motor_eff_neg_map)

# Need to read from CAN in real time. Set to inf in simulation
VeELSR_kW_MaxDischPwrElts = np.inf
VeELSR_kW_MaxChPwrElts = -np.inf

#%% Breton vehicle parameters
# With gear optimization
i0 = 4.267  # Final reduction drive ratio
eff_diff = 0.9  # Differential efficiency
eff_cpling = 0.9  # Coupling efficiency
m = 11800  # Empty load mass
r = 0.515  # Tyre radius; m
g = 9.8
grade = 0 / 180 * math.pi  # unit: rad
mu = 0.012  # Rolling resistance coefficient
air_density = 1.2258
C_A = 7.93  # Air area
C_d = 0.45  # Air resistance coefficient
rot_coef = 1.16  # Correction coefficient of rotating mass
gears = np.array([15.49, 12.07, 8.35, 5.67, 4.07, 2.96, 2.05, 1.39, 1.00])
Reg_rate = 0.3  #  Regeneration coefficient
transmission_speed_max = 2600 * (2 * np.pi / 60) # rpm
transmission_torque_max = 1800  # Nm
# gears_in_use = np.array([12.07, 8.35, 4.07, 1.39])  # 4 constantly used gears
gears_in_use = gears

def vel_preprocessing(vel_current, vel_pred, max_acc=2, max_dec=-2, t_delta=1):
	for i, v in enumerate(vel_pred):
		if i == 0:
			vel_pred[i] = max(vel_current + max_dec * t_delta, min(vel_current + max_acc * t_delta, v))
		else:
			vel_pred[i] = max(vel_pred[i - 1] + max_dec * t_delta, min(vel_pred[i - 1] + max_acc * t_delta, v))	
	return vel_pred

def vel_calc(torque, vel_current, gear):
	torque = np.array(torque)
	acc = (torque * i0 * eff_diff * eff_cpling * gear / r - m * g * mu - 0.5 * air_density * C_A * C_d * vel_current ** 2) / (rot_coef * m)
	vel_next = vel_current + acc
	return vel_next

def torque_wheel_calc(vel_now, vel_next, gear):  # velocity unit: m/s^2
	# Parameters
	i0 = 4.267  # Final reduction drive ratio
	eff_diff = 0.9  # Differential efficiency
	eff_cpling = 0.9  # Coupling efficiency
	m = 11800  # Empty load mass
	r = 0.515  # Tyre radius
	g = 9.8
	grade = 0 / 180 * math.pi  # unit: rad
	mu = 0.012  # Rolling resistance coefficient
	air_density = 1.2258
	C_A = 7.93  # Air area
	C_d = 0.45  # Air resistance coefficient
	rot_coef = 1.16  # Correction coefficient of rotating mass
	
	vel_mean = (vel_next + vel_now) / 2
	acc = vel_next - vel_now  # m/s^2
	F_slope_and_rolling = m * g * (np.sin(grade) + mu * np.cos(grade))
	F_air = 0.5 * air_density * C_A * C_d * vel_mean ** 2
	torque_I = (F_air + F_slope_and_rolling + m * acc * rot_coef) * r

	return torque_I

def torque_calc(vel_now, vel_next, gear):  # velocity unit: m/s^2
	# Parameters
	i0 = 4.267  # Final reduction drive ratio
	eff_diff = 0.9  # Differential efficiency
	eff_cpling = 0.9  # Coupling efficiency
	m = 11800  # Empty load mass
	r = 0.515  # Tyre radius
	g = 9.8
	grade = 0 / 180 * math.pi  # unit: rad
	mu = 0.012  # Rolling resistance coefficient
	air_density = 1.2258
	C_A = 7.93  # Air area
	C_d = 0.45  # Air resistance coefficient
	rot_coef = 1.16  # Correction coefficient of rotating mass
	
	vel_mean = (vel_next + vel_now) / 2
	acc = vel_next - vel_now  # m/s^2
	F_slope_and_rolling = m * g * (np.sin(grade) + mu * np.cos(grade))
	F_air = 0.5 * air_density * C_A * C_d * vel_mean ** 2
	torque = (F_air + F_slope_and_rolling + m * acc * rot_coef) * r / (i0 * eff_diff * eff_cpling * gear)  # Nm
	motor_speed = vel_mean * i0 * gear / r  # rad/s

	return (motor_speed, torque, acc)

def distance_calc(vel_current, vel_next):
    vel_mean = (vel_current + vel_next)/2
    return  vel_mean

def energy_and_motor_eff_calc(vel_seq, gear_seq, Reg_rate=Reg_rate, per_meter=True):  # vel_seq: ndarray; unit: m/s^2
	'''
	Calculate one or multi-step energy, motor efficiency and torque
	'''

	energy = 0
	motor_eff_seq = np.zeros(vel_seq.size - 1)
	torque_seq = np.zeros(vel_seq.size - 1)
	flag = 1
	if (vel_seq == 0).all():
		return 0, [0], [0], 1 

	# calculate one step energy
	if vel_seq.size == 2:
		(motor_speed, torque, acc) = torque_calc(vel_seq[0], vel_seq[1], gear_seq[0])
		s_delta = (vel_seq[0] + vel_seq[1]) / 2
		
		if motor_speed > min(transmission_speed_max, motor_pos_speeds.max()):#降档，print从*降到*、车速、加速度、
			gear_former =  gear_seq[0]
			while motor_speed > min(transmission_speed_max, motor_pos_speeds.max()):
				gear_seq[0] = gears_in_use[np.argwhere(gears_in_use == gear_seq[0]) + 1]
				motor_speed, _, _ = torque_calc(vel_seq[0], vel_seq[1], gear_seq[0])
			print(f'\noriginal motor speed exceed limits, change gear {gear_former} to {gear_seq[0]}\n'
				  f'velocity: {vel_seq[0], vel_seq[1]}\n'
				  f'acc: {vel_seq[1] - vel_seq[0]} m/s\n')
			# delta_speed = min(transmission_speed_max, motor_pos_speeds.max()) - motor_speed
			# motor_speed = min(transmission_speed_max, motor_pos_speeds.max())
			# print(f'\noriginal motor speed exceed limits, set to largest.\n'
			# 	f'velocity: {vel_seq[0], vel_seq[1]}\n'
			# 	f'delta motor speed: {delta_speed}\n'
			# 	f'acc: {vel_seq[1] - vel_seq[0]} m/s\n'
			# 	f'original motor_speed: {motor_speed}\n'
			# 	f'gear: {gear_seq[0]}\n')
		Tm_max = interpolate_pos_torque(motor_speed)
		Tm_min = interpolate_neg_torque(motor_speed)

		if torque >= 0:
			if torque > Tm_max:
				torque = Tm_max
				print(f'\noriginal torque exceeds limits, set to largest!')
			motor_eff_seq[0] = interpolate_pos_motor_eff(motor_speed, torque)
			energy = torque * motor_speed / motor_eff_seq[0] / s_delta if per_meter == True else torque * motor_speed / motor_eff_seq[0]
		else:
			if torque < Tm_min:
				torque = Tm_min
				print(f'\noriginal torque exceeds limits, set to largest!')
			motor_eff_seq[0] = interpolate_neg_motor_eff(motor_speed, torque)
			energy = torque * motor_speed * motor_eff_seq[0] * Reg_rate / s_delta if per_meter == True else torque * motor_speed * motor_eff_seq[0] * Reg_rate
			torque *= Reg_rate

		torque_seq[0] = torque
	else:
		for i in range(vel_seq.size - 1):
			vel_now = vel_seq[i]
			vel_next = vel_seq[i + 1]
			s_delta = (vel_now + vel_next) / 2
			(motor_speed, torque, acc) = torque_calc(vel_now, vel_next, gear_seq[i])
			if motor_speed > min(transmission_speed_max, motor_pos_speeds.max()):
				delta_speed = min(transmission_speed_max, motor_pos_speeds.max()) - motor_speed
				motor_speed = min(transmission_speed_max, motor_pos_speeds.max())
				flag = 0
				print(f'\nmulti-step: original motor speed exceed limits, set to largest.\n'
					f'delta motor speed: {delta_speed}\n'
					f'acc: {vel_seq[1] - vel_seq[0]} m/s\n'
					f'original motor_speed: {motor_speed}\n'
					f'gear: {gear_seq[0]}\n')
			Tm_max = interpolate_pos_torque(motor_speed)
			Tm_min = interpolate_neg_torque(motor_speed)
			if torque >= 0:
				if torque > Tm_max:
					torque = Tm_max
					print(f'\noriginal torque exceeds limits, set to largest!')
				motor_eff = interpolate_pos_motor_eff(motor_speed, torque)
				energy_temp = torque * motor_speed / motor_eff / s_delta if per_meter == True else torque * motor_speed / motor_eff
			else:
				if torque < Tm_min:
					torque = Tm_min
				motor_eff = interpolate_neg_motor_eff(motor_speed, torque)
				energy_temp = torque * motor_speed * motor_eff * Reg_rate / s_delta if per_meter == True else torque * motor_speed * motor_eff * Reg_rate
				torque *= Reg_rate
			energy += energy_temp
			motor_eff_seq[i] = motor_eff
			torque_seq[i] = torque
	return energy, torque_seq, motor_eff_seq, flag

def check_vel_tm_consistence(vel_seq, gear_opt, Tm_seq, Reg_rate=Reg_rate):
	vel_seq /= 3.6  # m/s
	flag = np.zeros(len(Tm_seq))
	for i in range(vel_seq.size-1):
		vel_now = vel_seq[i]
		vel_next = vel_seq[i+1]
		(motor_speed, torque, acc) = torque_calc(vel_now, vel_next, gear_opt[i])
		torque = torque * Reg_rate if torque < 0 else torque
		if np.isclose(torque, Tm_seq[i]):
			flag[i] = 1
	return flag


# %%
