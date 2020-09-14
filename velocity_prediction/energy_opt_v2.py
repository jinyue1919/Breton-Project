import matplotlib.pyplot as plt
from util_energy_opt import *

def energy_opt_v2(vel_current, vel_pred, gear_pre, tm_now=None, gear_pre_duration=3, delta_t=1, backward_sim=True):

	# vel_current = 24 / 3.6
	# vel_pred = np.ones(10) * 27 / 3.6
	# gear_pre: index of gears, range: 1-9
	# gear_pre_duration = 1  # at least 1 s
	# tm_now: current motor torque
	# tdmnd_cur: driver desired torque 
	# tm_now = 200
	# delta_t = 1  # s

	# vel_num_per_second =  6
	prediction_time_steps = vel_pred.size

	vel_pred = vel_preprocessing(vel_current, vel_pred)

	vel_max_rate = 1.1
	vel_min_rate = 0.9
	min_range = 2  # m/s

	vel_max = np.minimum(40, np.maximum(vel_pred * vel_max_rate, vel_pred + min_range))
	vel_min = np.maximum(0, np.minimum(vel_pred * vel_min_rate, vel_pred - min_range))

	# to match driver intension, the velocity change at first step shouldn't be too large
	vel_max[0] = min(vel_pred[0] + 0.5, vel_max[0])
	vel_min[0] = max(vel_pred[0] - 0.5, vel_min[0])
	t_axis = np.arange(1, prediction_time_steps + 1)

	# vel opt calculation
	# vel_max = vel_max.reshape(prediction_time_steps, 1)
	# vel_min = vel_min.reshape(prediction_time_steps, 1)
	vel_goal_end = vel_pred[-1]  # or vel_min[-1] as dp results suggest. set as vel_pred[-1] will make vel_opt approaches vel_pred
	vel_temp = vel_current
	t_temp = 0
	t_series = [t_temp]
	v_series = [vel_temp]
	flag = 1

	# search for 
	while flag:
		lin_delta_v = (vel_goal_end - vel_temp) / (prediction_time_steps - t_temp)  # or prediction_time_steps
		v_max = (vel_max - vel_temp) / np.arange(1 - t_temp, 1 - t_temp + prediction_time_steps) - lin_delta_v
		v_min = (vel_min - vel_temp) / np.arange(1 - t_temp, 1 - t_temp + prediction_time_steps) - lin_delta_v

		index = np.argwhere(np.logical_and(np.logical_or(v_max <= 0, v_min >= 0), t_axis > t_temp))

		# no strategy
		if index.size == 0:
			index = np.array([prediction_time_steps])

		# no strategy or reach the end of searching
		# acceleration is constant for the remaining time steps at (vel_goal_end - vel_temp) / (prediction_time_steps - t_temp)
		if index[0] == prediction_time_steps:
			vel_temp = vel_goal_end
			t_temp = prediction_time_steps
			t_series.append(prediction_time_steps)
			v_series.append(vel_goal_end)
			break
		else:
			if v_max[index[0]] <= 0:
				vel_temp1 = vel_max[index[0]]
			else:
				vel_temp1 = vel_min[index[0]]

			# find better route
			t_temp1 = t_axis[index[0]]
			lin_delta_v = (vel_temp1 - vel_temp) / (t_temp1 - t_temp)  # smaller
			v_max = (vel_max - vel_temp) / np.arange(1 - t_temp, 1 - t_temp + prediction_time_steps) - lin_delta_v
			v_min = (vel_min - vel_temp) / np.arange(1 - t_temp, 1 - t_temp + prediction_time_steps) - lin_delta_v

			index1 = np.argwhere(np.logical_and(np.logical_or(v_max <= 0, v_min >= 0), t_axis > t_temp))

			# no better route
			if index1.size == 0 or index1[0] >= index[0]:
				t_temp = t_temp1
				vel_temp = vel_temp1
			else:
				# better route found, reset vel_temp and t_temp, enter next iteration
				if v_max[index1[0]] <= 0:
					vel_temp = vel_max[index1[0]]
				else:
					vel_temp = vel_min[index1[0]]
				
				t_temp = t_axis[index1[0]]
			
			t_series.append(t_temp)
			v_series.append(vel_temp)

	# simplify the found route
	# i1 = 1
	# i2 = 1
	# if v_series.size > 2:
	#     flag = 1
	#     i1 = 1
	#     i2 = 3
	#     index_simp = i1
	# else:
	#     flag = 0
	#     index_simp = [i for i in range(1, v_series.size + 1)]

	# while flag:
	#     v1 = v_series[i1]
	#     t1 = t_series[i1]
	#     v2 = v_series[i2]
	#     t2 = t_series[i2]

	#     c = (v2 - v1) / (t2 - t1)
	#     a = (vel_max - v1) / np.arange(1 - t1, 1 - t1 + prediction_time_steps) - c
	#     b = (vel_min - v1) / np.arange(1 - t1, 1 - t1 + prediction_time_steps) - c

	#     index = np.argwhere(np.logical_and(np.logical_or(a <= 0, b >= 0), t_axis >= t1, t_axis < t2))

	vel_opt = np.zeros(prediction_time_steps + 1)
	for i in range(len(v_series) - 1):
		mean_acc = (v_series[i + 1] - v_series[i]) / (t_series[i + 1] - t_series[i])
		mean_vs = (v_series[i + 1] + v_series[i]) / 2
		time_len = t_series[i + 1] - t_series[i]
		if time_len >= 2 and mean_acc > 0:
			coef = -3.405 * mean_acc ** 2 + 0.6559 * mean_acc + 1.98
			coef = min(1.8, max(0.5, coef))
		elif time_len >= 2 and mean_acc < -0.3:
			coef = 0.441 * mean_acc ** 2 + 0.034 * mean_acc + 0.5854
			coef = min(1.8, max(0.5, coef))
		else:
			coef = 1
		t = np.arange(t_series[i], t_series[i + 1] + 1)
		vel_opt[t] = v_series[i] + mean_acc * time_len * ((t - t[0]) / time_len) ** coef

	vel_now = vel_opt[0]
	vel_next = vel_opt[1]
	acc = vel_next - vel_now
	vel_mean = (vel_now + vel_next) / 2.

	# gear_pre_duration is elapsed time since previous switching gears
	# in backward simulation, set gear_pre_duration to be > 2 always
	if gear_pre_duration < 2:
		# prevent switch gears too constantly, next step gear is the same as now
		gr = gear_pre
		motor_speed, torque, acc = torque_calc(vel_now, vel_next, gears_in_use[int(gr - 1)])
		Tm_max = min(interpolate_pos_torque(motor_speed), transmission_torque_max)
		Tm_min = max(interpolate_neg_torque(motor_speed), -transmission_torque_max)
		torque *= Reg_rate if torque < 0 else torque
		Tm_cur = min(Tm_max, max(Tm_min, torque))
		gear_cur = gr
	else:
		# switch gears
		gr_min = max(gear_pre - 2 - 1, 0)
		gr_max = min(gear_pre + 2 - 1, gears_in_use.size - 1)
		# length = gr_max - gr_min + 1
		length = len(gears_in_use)
		gr_temp = np.zeros(length)
		Tm_temp = np.zeros(length)
		power_temp = np.zeros(length)
		motor_eff_temp = np.zeros(length)
		
		# for count, gr in enumerate(np.arange(gr_min, gr_max + 1)):
		# 	gr_temp[count] = gears_in_use[int(gr - 1)]
		# 	motor_speed, torque, acc = torque_calc(vel_now, vel_next, gears_in_use[int(gr - 1)])
		for count, gr in enumerate(gears_in_use):
			gr_temp[count] = gr
			motor_speed, torque, acc = torque_calc(vel_now, vel_next, gr)

			if motor_speed > min(transmission_speed_max, motor_pos_speeds.max()):
				# next step torque is the same as now
				Tm_temp[count] = tm_now
				power_temp[count] = 10000000
				motor_eff_temp[count] = -1
			else:
				Tm_max = min(interpolate_pos_torque(motor_speed), transmission_torque_max)
				Tm_min = max(interpolate_neg_torque(motor_speed), -transmission_torque_max)
				
				if torque >= 0 and torque < Tm_max:
					motor_eff = interpolate_pos_motor_eff(motor_speed, torque)
					Tm_temp[count] = torque
					power_temp[count] = torque * motor_speed / motor_eff
					motor_eff_temp[count] = motor_eff
				elif torque < 0 and torque > Tm_min:
					motor_eff = interpolate_neg_motor_eff(motor_speed, torque)
					Tm_temp[count] = torque * Reg_rate
					power_temp[count] = torque * motor_speed * Reg_rate * motor_eff
					motor_eff_temp[count] = motor_eff
				else:
					Tm_temp[count] = min(Tm_max, max(torque, Tm_min))
					power_temp[count] = 10000000
					motor_eff_temp[count] = -1

		
		idx_min = np.argmin(power_temp)
		power_min = power_temp.min()

		if power_min == 10000000:
			# no strategy, don't optimize, output last step gear and Tm (original in backward simulation)
			gear_cur = gear_pre
			Tm_cur = tm_now / gears_in_use[gear_cur]
			flag = 0
			# print(f'No strategy, output previous torque and gear')
			print(f'No strategy, don\'t optimize')
		else:
			flag = 1
			gear_cur = gr_temp[idx_min]
			Tm_cur = Tm_temp[idx_min]
			motor_eff_cur = motor_eff_temp[idx_min]

	if backward_sim == False:
		# final output
		tm_rate = 1000 * 0.5  # Nm/s
		Tm_cur = max(min(tm_now + tm_rate * delta_t, Tm_cur), tm_now - tm_rate * delta_t)
	return vel_opt*3.6, vel_min*3.6, vel_max*3.6, Tm_cur, np.array([gear_cur]), motor_eff_cur, flag

if __name__ == '__main__':
	vel_current = 25 / 3.6
	vel_pred = np.array([26,28,25,26,27,29,30,31.2,32.1,33.7])  / 3.6
	gear_pre = 7  # 1-9
	_, tm_now, _ = torque_calc(vel_current, vel_current, gears_in_use[int(gear_pre - 1)])
	vel_opt, vel_min, vel_max, Tm_cur, gear_cur, motor_eff_cur, flag = energy_opt_v2(vel_current, vel_pred, gear_pre)
	