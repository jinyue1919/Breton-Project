import matplotlib.pyplot as plt
from util_energy_opt import *
import copy

#TODO: may need to consider driver intention (if the driver hits the acceleration pedal, he intents to accelerate which needs the motor
# to output positive torque, but current optimized torque in the next step may be negative, there is no limits on this.
# Ideally, the predicted velocity represents driver intention, i.e. to reach some velocity after sometime. But the predicted
# veloctity includes error. Plus, the optimized velocity may well go astray from the original driver intention -> TODO: when measure
# the validaty of velocity prediction module, measure how often does it reflect driver intention)
# and torque difference limit betwee two steps
def energy_opt_v1(gears_in_use, gear_ctl, vel_current, vel_pred, gear_seq=None, vel_min_rate=0.9, vel_max_rate=1.1, vel_num_per_second=6, t_delta=1, vel_min_range=1, max_acc=2, max_dec=-2, gearOpt=False):
	# vel_current, vel_pred: m/s
	gear_index = (np.where(gears_in_use == gear_ctl)[0])[0]
	if gear_index == 0:
		gears_in_use = copy.deepcopy(gears_in_use[gear_index: gear_index +2])
	elif gear_index == len(gears_in_use) - 1:
		gears_in_use = copy.deepcopy(gears_in_use[gear_index-2: gear_index+2])
	elif gear_index == 1:
		gears_in_use = copy.deepcopy(gears_in_use[0: gear_index+2])
	elif gear_index == len(gears_in_use) - 2:
		gears_in_use = copy.deepcopy(gears_in_use[gear_index - 2: len(gears_in_use)-2])
	else:
		gears_in_use = copy.deepcopy(gears_in_use[gear_index - 2: gear_index + 2])
	
	prediction_time_steps = vel_pred.size

	vel_pred_r = vel_preprocessing(vel_current, vel_pred, max_acc, max_dec)
	vel_pred_r = vel_pred_r.reshape(prediction_time_steps, 1)

	vel_max = np.minimum(40, np.maximum(vel_pred_r * vel_max_rate, vel_pred_r + vel_min_range))
	vel_min = np.maximum(0, np.minimum(vel_pred_r * vel_min_rate, vel_pred_r - vel_min_range))
	vel_max[0] = min(vel_pred[0] + 0.5, vel_max[0])
	vel_min[0] = max(vel_pred[0] - 0.5, vel_min[0])

	vel_pred_mat = np.zeros((prediction_time_steps, vel_num_per_second))
	for i, (v_max, v_min) in enumerate(zip(vel_max, vel_min)):
		vel_pred_mat[i] = np.linspace(v_min, v_max, vel_num_per_second).squeeze(1)
	vel_mat =  np.concatenate([vel_current * np.ones([1, vel_num_per_second]),
		vel_pred_mat], axis=0)

	time_steps_total = prediction_time_steps + 1

	assert list(vel_mat.shape) == [time_steps_total, vel_num_per_second], 'Wrong vel_mat build!'

	if gearOpt == True:
		# With gear optimization
		state_num = vel_num_per_second ** 2 * gears_in_use.size
	else:
		# No gear optimization
		state_num = vel_num_per_second ** 2

	# Cost matrix Pb[i, j]: power from (i-1)-th time step to i-th time step, at j-th ${state_count}, etc.
	# Note: first row of Pb, Acc, Motor_eff, Sys_mode and Tm contains only zeros TODO: optimize cost matrix
	Pb = np.zeros([prediction_time_steps, state_num])
	Acc = np.zeros([prediction_time_steps, state_num])
	Motor_eff = np.zeros([prediction_time_steps, state_num])
	Sys_mode = np.ones([prediction_time_steps, state_num])
	Tm = np.zeros([prediction_time_steps, state_num])

	# Construct cost matrix
	# No gear optimization
	if gearOpt == False:
		for t in range(1, time_steps_total):
			state_count = 0
			gear_current = gear_seq[t - 1]
			for current_state in range(vel_num_per_second):
				vel_now = vel_mat[t, current_state]
				for last_state in range(vel_num_per_second):
					vel_last = vel_mat[t - 1, last_state]
					s_delta = (vel_last + vel_now) / 2 * t_delta
					# TODO: Add transmission map. Use the original gear sequence tempe
					(motor_speed, torque, acc) = motor_torque_calc(vel_last, vel_now, gear=gear_current)
					
					if motor_speed > min(transmission_speed_max, motor_pos_speeds.max(), motor_neg_speeds.max()) or acc > max_acc or acc < max_dec:
						power = np.inf
						sys_mode = 0
					else:
						Tm_max = interpolate_pos_torque(motor_speed)
						Tm_min = interpolate_neg_torque(motor_speed)
					
						if (torque >= 0) and (torque < min(Tm_max, transmission_torque_max)):
							motor_eff = interpolate_pos_motor_eff(motor_speed, torque)
							power = torque * motor_speed / motor_eff / s_delta
							sys_mode = 1
							# Final battery discharging power limit after considering BMS
							# discharging power limit, current SOC and auxiliary power limit
							if (power > VeELSR_kW_MaxDischPwrElts):
								power = np.inf
								sys_mode = 0
						elif (torque < 0) and (torque > max(Tm_min, -transmission_torque_max)):
							motor_eff = interpolate_neg_motor_eff(motor_speed, torque)
							power = torque * motor_speed * motor_eff / s_delta
							torque *= Reg_rate
							sys_mode = 1
							# Final battery charging power limit after considering BMS
							# charging power limit, current SOC and auxiliary power limit
							if (power < VeELSR_kW_MaxChPwrElts):
								power = np.inf
								sys_mode = 0
						else:
							sys_mode = 0
							motor_eff = 0
							power = np.inf
					
					# update cost matrix
					Acc[t, state_count] = acc
					Pb[t, state_count] = power  # w/m
					Tm[t, state_count] = torque
					Sys_mode[t, state_count] = sys_mode
					Motor_eff[t, state_count] = motor_eff
					
					state_count += 1
	else:
		# With gear optimization
		for t in range(0, prediction_time_steps):
			state_count = 0
			for gear in gears_in_use:
				for state_next in range(vel_num_per_second):
					vel_next = vel_mat[t+1, state_next]
					for state_now in range(vel_num_per_second):
						vel_now = vel_mat[t, state_now]
						s_delta = (vel_now + vel_next) / 2 * t_delta  # distance between two timestep -> for calculating cost (w/m)
						(motor_speed, torque, acc) = motor_torque_calc(vel_now, vel_next, gear=gear)
						# check if motor speed is output limits
						sys_mode_temp = 1
						if motor_speed > min(transmission_speed_max, motor_pos_speeds.max()) or acc > max_acc or acc < max_dec:
							power = 10000000
							sys_mode_temp = 0
							motor_eff = 0.000000001
						else:
							Tm_max = interpolate_pos_torque(motor_speed)
							Tm_min = interpolate_neg_torque(motor_speed)
							
							if (torque >= 0) and (torque < min(Tm_max, transmission_torque_max)):
								motor_eff = interpolate_pos_motor_eff(motor_speed, torque)
								power = torque * motor_speed / motor_eff / s_delta
								# Final battery discharging power limit after considering BMS
								# discharging power limit, current SOC and auxiliary power limit
								if (power > VeELSR_kW_MaxDischPwrElts) or np.isnan(power) :
									power = 10000000
									sys_mode_temp = 0
							
							elif (torque < 0) and (torque > max(Tm_min, -transmission_torque_max)):
								motor_eff = interpolate_neg_motor_eff(motor_speed, torque)
								power = torque * motor_speed * motor_eff / s_delta * Reg_rate
								torque *= Reg_rate
								# Final battery charging power limit after considering BMS
								# charging power limit, current SOC and auxiliary power limit
								if (power < VeELSR_kW_MaxChPwrElts) or np.isnan(power):
									power = 10000000
									sys_mode_temp = 0
							else:
								motor_eff = 0.000000001
								sys_mode_temp = 0
								power = 10000000
						
						# update cost matrix
						Acc[t, state_count] = acc
						Pb[t, state_count] = power  # w*s/m
						Tm[t, state_count] = torque
						Sys_mode[t, state_count] = sys_mode_temp
						Motor_eff[t, state_count] = motor_eff
						state_count += 1

	# Cost to go
	Jcost = np.zeros([2, state_num])
	vel_opt = np.zeros(prediction_time_steps+1)
	acc_opt = np.zeros(prediction_time_steps)
	sys_mode_opt = np.zeros(prediction_time_steps) 
	Tm_opt = np.zeros(prediction_time_steps)
	motor_eff_opt = np.zeros(prediction_time_steps)
	route = np.zeros([prediction_time_steps, state_num])
	route_opt = np.zeros(prediction_time_steps)

	for t in range(prediction_time_steps):
		for state_count in range(state_num):
			if gearOpt == False:
				bgn = np.mod(state_count, vel_num_per_second) * vel_num_per_second
				end = bgn + vel_num_per_second
				route[t, state_count] = np.argmin(Jcost[0, bgn:end] + Pb[t, state_count]) + bgn
				Jcost[1, state_count] = np.amin(Jcost[0, bgn:end] + Pb[t, state_count])
			else:
				delta = 0
				JcostTemp = {}
				for i in range(len(gears_in_use)):
					bgn = np.mod(state_count, vel_num_per_second) * vel_num_per_second + delta
					end = bgn + vel_num_per_second
					if Sys_mode[t, stamotor_torque_calc= 1:
						idx = np.argmin(Jcost[0, bgn:end] + Pb[t, state_count]) + bgn
						cost = np.amin(Jcost[0, bgn:end] + Pb[t, state_count])
					elif Sys_mode[t,state_count] == 0:
						idx = 0
						cost = 10000000

					JcostTemp[idx] = cost
					delta += vel_num_per_second ** 2
				
				indexes_and_costs = sorted(JcostTemp.items(), key=lambda x: x[1])

				route[t, state_count] = indexes_and_costs[0][0]
				Jcost[1, state_count] = indexes_and_costs[0][1]
		
		Jcost[0, :] = Jcost[1, :]

	# Save minimum total cost and index at last time step
	JcostMin = np.amin(Jcost[1, :])
	route_opt[prediction_time_steps-1] = np.argmin(Jcost[1, :])

	# route_opt[t]: the best state from (t-1)-th step to t-th step
	for t in range(prediction_time_steps-1, 0, -1):
		route_opt[t-1] = route[t, int(route_opt[t])]  # Calculate route_opt reversely

	gear_opt = []
	vel_opt[0] = vel_current
	for t in range(prediction_time_steps):
		if gearOpt == True:
			# With gear optimization
			vel_opt[t+1] = vel_mat[t+1, int(np.mod(route_opt[t], vel_num_per_second ** 2) / vel_num_per_second)]
			acc_opt[t] = Acc[t, int(route_opt[t])]
			gear_opt.append(gears_in_use[int(route_opt[t] / vel_num_per_second ** 2)])
			Tm_opt[t] = Tm[t, int(route_opt[t])]
			sys_mode_opt[t] = Sys_mode[t, int(route_opt[t])]
			motor_eff_opt[t] = Motor_eff[t, int(route_opt[t])]
		elif gearOpt == False:
			vel_opt[t] = vel_mat[t, int(route_opt[t] / vel_num_per_second)]

	power, torque_seq_check, motor_eff_check, flag_torque = energy_and_motor_eff_calc(vel_opt, gear_opt)
	if gearOpt == True:
		# if JcostMin == energy_calc(vel_opt, gear_opt, Reg_rate=0.1):
		if np.isclose(JcostMin, power) and np.allclose(motor_eff_opt, motor_eff_check):
			# print(JcostMin, energy_calc(vel_opt, gear_opt, Reg_rate=Reg_rate))
			flag = 1
		else:
			flag = 0
	elif gearOpt == False:
		power, _ = energy_and_motor_eff_calc(vel_opt, gear_seq=gear_seq)
		if JcostMin == power:
			flag = 1
		else:
			flag = 0
	# fig, ax = plt.subplots()
	# ax.plot(vel_pred.flatten(), label='pred')
	# ax.plot(vel_opt[1:], label='opt')
	# ax.plot(vel_min.flatten(), label='min')
	# ax.plot(vel_max.flatten(),label='max')
	# ax.legend()
	# plt.show()
	return flag, Tm_opt, vel_opt * 3.6, vel_min * 3.6, vel_max * 3.6, vel_pred_r, np.array(gear_opt), motor_eff_opt, JcostMin, sys_mode_opt

if __name__ == '__main__':
	vel_current = 27
	prediction_time_steps = 10
	vel_pred = np.ones([prediction_time_steps, 1]) * 27
	
	(flag, Tm_opt, vel_opt, vel_min, vel_max, vel_pred_r, gear_opt, motor_eff_opt, JcostMin, sys_mode_opt) = energy_opt_v1(vel_current/3.6, vel_pred/3.6, vel_num_per_second=10, gearOpt=True)
	
	fig, ax = plt.subplots()
	ax.plot(3.6*vel_pred_r.flatten(), label='pred')
	ax.plot(vel_opt[1:], label='opt')
	ax.plot(vel_min.flatten(), label='min')
	ax.plot(vel_max.flatten(),label='max')
	ax.legend()
	plt.show()

	flags = check_vel_tm_consistence(vel_opt, gear_opt, Tm_opt)
	assert flag == 1, 'energy_opt error'
	assert list(flags) == [1] * Tm_opt.size