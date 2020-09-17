import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def w_modify(T):
# 论域
# input
	T_delta = ctrl.Antecedent(np.linspace(-150, 150, 301), 'T_delta')
	# output
	w1 = ctrl.Consequent(np.linspace(0.8, 1.2, 41), 'opt_w')

	# 隶属度函数
	# input
	T_delta['NB'] = fuzz.trapmf(T_delta.universe, [-160, -155, -150, -75])
	T_delta['NS'] = fuzz.trimf(T_delta.universe, [-150, -75, 0])
	T_delta['Z'] = fuzz.trimf(T_delta.universe, [-75, 0, 75])
	T_delta['PS'] = fuzz.trimf(T_delta.universe, [0, 75, 150])
	T_delta['PB'] = fuzz.trapmf(T_delta.universe, [75, 150, 151, 152])

	# output
	w1['S'] = fuzz.trapmf(w1.universe, [0, 0.3, 0.8, 0.93])
	w1['MS'] = fuzz.trimf(w1.universe, [0.8, 0.93, 1.07])
	w1['MB'] = fuzz.trimf(w1.universe, [0.93, 1.07, 1.2])
	w1['B'] = fuzz.trapmf(w1.universe, [1.07, 1.2, 1.5, 1.6])


	# T_delta.view()
	# plt.xlabel('转矩偏差')
	# plt.legend(loc='upper right',bbox_to_anchor=(1.15, 1))
	# plt.show()

	# w1.view()
	# plt.xlabel('优化扭矩权重')
	# plt.legend(loc='upper right',bbox_to_anchor=(1.15, 1))
	# plt.show()

	# Define rule base
	rule1 = ctrl.Rule(T_delta['NB'], (w1['S']))

	rule2 = ctrl.Rule(T_delta['NS'], (w1['MB']))

	rule3 = ctrl.Rule(T_delta['Z'], (w1['B']))

	rule4 = ctrl.Rule(T_delta['PS'], (w1['MB']))

	rule5 = ctrl.Rule(T_delta['PB'], (w1['S']))


	# rule1.view()

	# Create control systems
	w_modify_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])

	w_modify_fc = ctrl.ControlSystemSimulation(w_modify_ctrl)

	w_modify_fc.input['T_delta'] = T

	# w_modify_fc.input['T_delta'] = 150

	w_modify_fc.compute()

	# print (w_modify_fc.output['opt_w'])
		# w1.view(sim=w_modify_fc)


		# print(driver_intention_fc.output['driver intention'])
		# driver_intention.view(sim=driver_intention_fc)
	return w_modify_fc.output['opt_w']
 
