from geneticalgorithm import geneticalgorithm as ga
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def w_modify(T,delta_T,a_intension):
	# 论域
	# input
	T_delta = ctrl.Antecedent(np.linspace(-150, 150, 301), 'T_delta')
	delta_T_delta = ctrl.Antecedent(np.linspace(-200, 200, 401), 'delta_T_delta')
	acc_intension = ctrl.Antecedent(np.linspace(0, 1, 11), 'acc_intension') 
	# output
	w1 = ctrl.Consequent(np.linspace(0.8, 1.2, 41), 'opt_w')

	# 隶属度函数
	# input
	T_delta['NB'] = fuzz.trapmf(T_delta.universe, [a1_1 - 2, a1_1 - 1, a1_1, b1_1])
	T_delta['NS'] = fuzz.trimf(T_delta.universe, [a1_2, c1_2, b1_2])
	T_delta['Z'] = fuzz.trimf(T_delta.universe, [a1_3, c1_3, b1_3])
	T_delta['PS'] = fuzz.trimf(T_delta.universe, [a1_4, c1_4, b1_4])
	T_delta['PB'] = fuzz.trapmf(T_delta.universe, [a1_5, b1_5, b1_5 + 1, b1_5 + 2])

	delta_T_delta['NB'] = fuzz.trapmf(delta_T_delta.universe, [a2_1 - 2, a2_1 - 1, a2_1, b2_1])
	delta_T_delta['NS'] = fuzz.trimf(delta_T_delta.universe, [a2_2, c2_2, b2_2])
	delta_T_delta['Z'] = fuzz.trimf(delta_T_delta.universe, [a2_3, c2_3, b2_3])
	delta_T_delta['PS'] = fuzz.trimf(delta_T_delta.universe, [a2_4, c2_4, b2_4])
	delta_T_delta['PB'] = fuzz.trapmf(delta_T_delta.universe, [a2_5, b2_5, b2_5 + 1, b2_5 +2])

	acc_intension['S'] = fuzz.trapmf(acc_intension.universe, [a3_1 - 2, a3_1 - 1, a3_1, b3_1])
	acc_intension['MS'] = fuzz.trimf(acc_intension.universe, [a3_2, c3_2, b3_2])
	acc_intension['MB'] = fuzz.trimf(acc_intension.universe, [a3_3, c3_3, b3_3])
	acc_intension['B'] = fuzz.trapmf(acc_intension.universe, [a3_4, b3_4, b3_4 + 1, b3_4 + 2])
	# output
	w1['S'] = fuzz.trapmf(w1.universe, [ay_1 - 2, ay_1 - 1, ay_1, by_1])
	w1['MS'] = fuzz.trimf(w1.universe, [ay_2, cy_2, by_2])
	w1['MB'] = fuzz.trimf(w1.universe, [ay_3, cy_3, by_3])
	w1['B'] = fuzz.trapmf(w1.universe, [ay_4, by_4, by_4 + 1, by_4 + 2])

		
	# T_delta.view()
	# plt.xlabel('转矩偏差')
	# plt.legend(loc='upper right',bbox_to_anchor=(1.15, 1))

	# delta_T_delta.view()
	# plt.legend(loc='upper right',bbox_to_anchor=(1.15, 1))
	# plt.xlabel('转矩趋势偏差')

	# acc_intension.view()
	# plt.legend(loc='upper right',bbox_to_anchor=(1.15, 1))
	# plt.xlabel('加速意图')

	# w1.view()
	# plt.legend(loc='upper right',bbox_to_anchor=(1.15, 1))
	# plt.xlabel('优化扭矩权重')

	# plt.legend(loc='upper right',bbox_to_anchor=(1.15, 1))
	# plt.xlabel('需求扭矩权重')

	# Define rule base
	rule1 = ctrl.Rule(acc_intension['B'] & T_delta['NB'] & (delta_T_delta['PB'] | delta_T_delta['PS'] | delta_T_delta['NS'] | delta_T_delta['NB']), (w1['S']))
	rule2 = ctrl.Rule(acc_intension['B'] & T_delta['NB'] & delta_T_delta['Z'], (w1['MS']))

	rule3 = ctrl.Rule(acc_intension['B'] & T_delta['NS'] & (delta_T_delta['PB'] | delta_T_delta['NB']), (w1['S']))
	rule4 = ctrl.Rule(acc_intension['B'] & T_delta['NS'] & delta_T_delta['Z'], (w1['MS']))
	rule5 = ctrl.Rule(acc_intension['B'] & T_delta['NS'] & (delta_T_delta['NS'] | delta_T_delta['PS']) , (w1['MS']))

	rule6 = ctrl.Rule(acc_intension['B'] & T_delta['Z'] & (delta_T_delta['Z'] | delta_T_delta['NS'] | delta_T_delta['PS']), (w1['MS']))
	rule7 = ctrl.Rule(acc_intension['B'] & T_delta['Z'] & (delta_T_delta['PB'] | delta_T_delta['NB']), (w1['MS']))

	rule8 = ctrl.Rule(acc_intension['B'] & T_delta['PS'] & (delta_T_delta['NB'] | delta_T_delta['PB']) , (w1['MS']))
	rule9 = ctrl.Rule(acc_intension['B'] & T_delta['PS'] & (delta_T_delta['PS'] | delta_T_delta['NS']), (w1['MB']))
	rule10 = ctrl.Rule(acc_intension['B'] & T_delta['PS'] & delta_T_delta['Z'],(w1['MB']))

	rule11 = ctrl.Rule(acc_intension['B'] & T_delta['PB'] & (delta_T_delta['NB'] | delta_T_delta['NS'] | delta_T_delta['PS'] | delta_T_delta['PB']), (w1['MS']))
	rule12 = ctrl.Rule(acc_intension['B'] & T_delta['PB'] & delta_T_delta['Z'], (w1['MB']))

	rule13 = ctrl.Rule(acc_intension['S'] & T_delta['NB'] & (delta_T_delta['PS'] | delta_T_delta['PB'] | delta_T_delta['NB'] | delta_T_delta['NS']), (w1['B']))
	rule14 = ctrl.Rule(acc_intension['S'] & T_delta['NB'] & delta_T_delta['Z'],(w1['MB']))  

	rule15 = ctrl.Rule(acc_intension['S'] & T_delta['NS'] & (delta_T_delta['PB'] | delta_T_delta['NB']) , (w1['B']))
	rule16 = ctrl.Rule(acc_intension['S'] & T_delta['NS'] & (delta_T_delta['Z'] | delta_T_delta['PS'] | delta_T_delta['NS']), (w1['B']))  

	rule17 = ctrl.Rule(acc_intension['S'] & T_delta['Z'] & (delta_T_delta['PS'] | delta_T_delta['Z'] | delta_T_delta['NS']), (w1['B']))
	rule18 = ctrl.Rule(acc_intension['S'] & T_delta['Z'] & (delta_T_delta['NB'] | delta_T_delta['PB']), (w1['MB']))

	rule19 = ctrl.Rule(acc_intension['S'] & T_delta['PS'] & (delta_T_delta['NB'] | delta_T_delta['PB']) , (w1['MB']))
	rule20 = ctrl.Rule(acc_intension['S'] & T_delta['PS'] & (delta_T_delta['PS'] | delta_T_delta['Z'] | delta_T_delta['NS']) , (w1['B']))

	rule21 = ctrl.Rule(acc_intension['S'] & T_delta['PB'] & delta_T_delta['NB'], (w1['MB']))
	rule22 = ctrl.Rule(acc_intension['S'] & T_delta['PB'] & (delta_T_delta['Z'] | delta_T_delta['PS'] | delta_T_delta['NS']), (w1['B']))
	rule23 = ctrl.Rule(acc_intension['S'] & T_delta['PB'] & delta_T_delta['PB'], (w1['MB']))

	rule24 = ctrl.Rule(acc_intension['MS'] & T_delta['NB'] & (delta_T_delta['PB'] | delta_T_delta['NB']) , (w1['MS']))
	rule25 = ctrl.Rule(acc_intension['MS'] & T_delta['NB'] & delta_T_delta['Z'] , (w1['MB']))
	rule26 = ctrl.Rule(acc_intension['MS'] & T_delta['NB'] & (delta_T_delta['PS'] | delta_T_delta['NS']), (w1['MB']))

	rule27 = ctrl.Rule(acc_intension['MS'] & T_delta['NS'] & (delta_T_delta['PB'] | delta_T_delta['NB']), (w1['MS']))
	rule28 = ctrl.Rule(acc_intension['MS'] & T_delta['NS'] & (delta_T_delta['Z'] | delta_T_delta['PS'] | delta_T_delta['NS']) , (w1['MB']))

	rule29 = ctrl.Rule(acc_intension['MS'] & T_delta['Z'] & (delta_T_delta['PS'] | delta_T_delta['Z'] | delta_T_delta['NS']), (w1['B']))
	rule30 = ctrl.Rule(acc_intension['MS'] & T_delta['Z'] & (delta_T_delta['NB'] | delta_T_delta['PB']), (w1['MB']))

	rule31 = ctrl.Rule(acc_intension['MS'] & T_delta['PS'] & (delta_T_delta['NB'] | delta_T_delta['PB']), (w1['MB']))
	rule32 = ctrl.Rule(acc_intension['MS'] & T_delta['PS'] & (delta_T_delta['NS'] | delta_T_delta['Z'] | delta_T_delta['PS']), (w1['B']))

	rule33 = ctrl.Rule(acc_intension['MS'] & T_delta['PB'] & (delta_T_delta['NB'] | delta_T_delta['NS'] | delta_T_delta['PS'] | delta_T_delta['PB']), (w1['MB']))
	rule34 = ctrl.Rule(acc_intension['MS'] & T_delta['PB'] & delta_T_delta['Z']   , (w1['B']))

	rule35 = ctrl.Rule(acc_intension['MB'] & T_delta['NB'] & (delta_T_delta['PB'] | delta_T_delta['NB']) , (w1['S']))
	rule36 = ctrl.Rule(acc_intension['MB'] & T_delta['NB'] & (delta_T_delta['PS'] | delta_T_delta['NS']), (w1['MS'] ))
	rule37 = ctrl.Rule(acc_intension['MB'] & T_delta['NB'] & delta_T_delta['Z'], (w1['MS']))

	rule38 = ctrl.Rule(acc_intension['MB'] & T_delta['NS'] & (delta_T_delta['PB'] | delta_T_delta['NB']), (w1['S']))
	rule39 = ctrl.Rule(acc_intension['MB'] & T_delta['NS'] & (delta_T_delta['PS'] | delta_T_delta['NS'] | delta_T_delta['Z']), (w1['MS']))

	rule40 = ctrl.Rule(acc_intension['MB'] & T_delta['Z'] & delta_T_delta['Z'] , (w1['MS']))
	rule41 = ctrl.Rule(acc_intension['MB'] & T_delta['Z'] & (delta_T_delta['NS'] | delta_T_delta['PS']) , (w1['MS']))
	rule42 = ctrl.Rule(acc_intension['MB'] & T_delta['Z'] & delta_T_delta['NB'] , (w1['MS']))
	rule43 = ctrl.Rule(acc_intension['MB'] & T_delta['Z'] & delta_T_delta['PB'] , (w1['MS']))

	rule44 = ctrl.Rule(acc_intension['MB'] & T_delta['PS'] & (delta_T_delta['NB'] | delta_T_delta['PB']), (w1['MS']))
	rule45 = ctrl.Rule(acc_intension['MB'] & T_delta['PS'] & (delta_T_delta['NS'] | delta_T_delta['Z'] | delta_T_delta['PS']), (w1['MB'])) 

	rule46 = ctrl.Rule(acc_intension['MB'] & T_delta['PB'] & delta_T_delta['Z'], (w1['MB']))
	rule47 = ctrl.Rule(acc_intension['MB'] & T_delta['PB'] & (delta_T_delta['NB'] | delta_T_delta['PB']), (w1['MS']))
	rule48 = ctrl.Rule(acc_intension['MB'] & T_delta['PB'] & (delta_T_delta['NS'] | delta_T_delta['PS']), (w1['MS']))

	# Create control systems
	w_modify_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9,
	rule10, rule11, rule12, rule12, rule13, rule14, rule15, rule16, rule16, rule17, rule18, rule19, rule20,
	rule21, rule22, rule23, rule24, rule25, rule26, rule27, rule28, rule29, rule30, rule31, rule32, rule33,
	rule34, rule35, rule36, rule37, rule38, rule39, rule40, rule41, rule42, rule43, rule44, rule45, rule46, rule47, rule48])

	w_modify_fc = ctrl.ControlSystemSimulation(w_modify_ctrl)

	w_modify_fc.input['T_delta'] = T
	w_modify_fc.input['delta_T_delta'] = delta_T
	w_modify_fc.input['acc_intension'] = a_intension

	w_modify_fc.compute()

	# print (w_modify_fc.output['opt_w'])
	# print (w_modify_fc.output['dmd_w'])
	# w1.view(sim=w_modify_fc)


	# print(driver_intention_fc.output['driver intention'])
	# driver_intention.view(sim=driver_intention_fc)
	return w_modify_fc.output['opt_w']

W1 = W1 * w1
torque_ctl = W1 * Tm_opt[0] + (1-W1) * torque_ori
vel_ctl = vel_calc(torque_ctl, vel_current / 3.6, gear_ctl) * 3.6
