import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

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
	T_delta['NB'] = fuzz.trapmf(T_delta.universe, [-160, -155, -150, -75])
	T_delta['NS'] = fuzz.trimf(T_delta.universe, [-150, -75, 0])
	T_delta['Z'] = fuzz.trimf(T_delta.universe, [-75, 0, 75])
	T_delta['PS'] = fuzz.trimf(T_delta.universe, [0, 75, 150])
	T_delta['PB'] = fuzz.trapmf(T_delta.universe, [75, 150, 151, 152])

	delta_T_delta['NB'] = fuzz.trapmf(delta_T_delta.universe, [-210, -205, -200, -100])
	delta_T_delta['NS'] = fuzz.trimf(delta_T_delta.universe, [-200, -100, 0])
	delta_T_delta['Z'] = fuzz.trimf(delta_T_delta.universe, [-100, 0, 100])
	delta_T_delta['PS'] = fuzz.trimf(delta_T_delta.universe, [0, 100, 200])
	delta_T_delta['PB'] = fuzz.trapmf(delta_T_delta.universe, [100, 200, 205, 210])

	acc_intension['S'] = fuzz.trapmf(acc_intension.universe, [-2, -1, 0, 0.3])
	acc_intension['MS'] = fuzz.trimf(acc_intension.universe, [0, 0.3, 0.6])
	acc_intension['MB'] = fuzz.trimf(acc_intension.universe, [0.3, 0.6, 1])
	acc_intension['B'] = fuzz.trapmf(acc_intension.universe, [0.6, 1, 1.1, 1.2])
	# output
	w1['S'] = fuzz.trapmf(w1.universe, [0, 0.3, 0.8, 0.93])
	w1['MS'] = fuzz.trimf(w1.universe, [0.8, 0.93, 1.07])
	w1['MB'] = fuzz.trimf(w1.universe, [0.93, 1.07, 1.2])
	w1['B'] = fuzz.trapmf(w1.universe, [1.07, 1.2, 1.5, 1.6])

		
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

	# rule1.view()

	# Create control systems
	w_modify_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9,
	rule10, rule11, rule12, rule12, rule13, rule14, rule15, rule16, rule16, rule17, rule18, rule19, rule20,
	rule21, rule22, rule23, rule24, rule25, rule26, rule27, rule28, rule29, rule30, rule31, rule32, rule33,
	rule34, rule35, rule36, rule37, rule38, rule39, rule40, rule41, rule42, rule43, rule44, rule45, rule46, rule47, rule48])

	w_modify_fc = ctrl.ControlSystemSimulation(w_modify_ctrl)

	w_modify_fc.input['T_delta'] = T
	w_modify_fc.input['delta_T_delta'] = delta_T
	w_modify_fc.input['acc_intension'] = a_intension

	# w_modify_fc.input['T_delta'] = -50
	# w_modify_fc.input['delta_T_delta'] = -90
	# w_modify_fc.input['acc_intension'] = 0.8

	w_modify_fc.compute()

	# print (w_modify_fc.output['opt_w'])
	# print (w_modify_fc.output['dmd_w'])
	# w1.view(sim=w_modify_fc)


	# print(driver_intention_fc.output['driver intention'])
	# driver_intention.view(sim=driver_intention_fc)
	return w_modify_fc.output['opt_w']
 
#  def f():
# 	 return 

# varbound = np.array([[-200,200] * 10, [-150,150] * 10, [0,1] * 6, [0.9,1.4] * 12])
# model = ga(function = f, dimension = 38, variable_type = 'real', variable_boundaries = varbound)