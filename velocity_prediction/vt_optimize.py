import numpy as np
from scipy.optimize import minimize
import pickle
import math
from scipy.optimize import Bounds
import matplotlib.pyplot as plt

# initial parameters
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
Reg_rate = 0.1  #  Regeneration coefficient
transmission_speed_max = 2600 * (2 * np.pi / 60) # rpm
transmission_torque_max = 1800  # Nm
eff_motor = 0.9 # assume eff_motor is a constant
# gears_in_use = np.array([12.07, 8.35, 4.07, 1.39])  # 4 constantly used gears
gears_in_use = gears

# calculate some constants
energy_const = 10 * m * g * (np.sin(grade) + mu * np.cos(grade)) * eff_motor * Reg_rate / (eff_cpling * eff_diff) 
F_a_const = 0.5 * air_density * C_A * C_d
energy_der_const = F_a_const * eff_motor * Reg_rate / (eff_cpling * eff_diff)

bounds = Bounds([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf, math.inf])

# load data
with open('preprocessing_data.pickle','rb') as f:
    [v_test, distance, v_km] = pickle.load(f)     # v_test : m/s  distance : m

# objective function
def energy(x):
    ''' energy consumption in 10 steps '''
    return sum(m * rot_coef * (x[1:] - x[:-1]) + F_a_const * 0.25 * (x[:-1] + x[1:]) ** 2)

# derivate objective function
def energy_der(x):
    der = np.zeros_like(x)
    der[0] = eff_motor * Reg_rate / (eff_cpling * eff_diff) * (F_a_const * 0.5 * (x[0] + x[1]) - m * rot_coef)
    der[1:-1] = eff_motor * Reg_rate / (eff_cpling * eff_diff) * F_a_const * 0.5 * (x[:-2] + 2 * x[1:-1] + x[2:])
    der[-1] = eff_motor * Reg_rate / (eff_cpling * eff_diff) * (F_a_const * 0.5 * (x[-1] + x[-2]) + m * rot_coef)
    return der


# define inequal constrains
ineq_cons = {'type': 'ineq',
             'fun' : lambda x: np.array([x[1] - x[0] + 2.5,
                                         x[2] - x[1] + 2.5,
                                         x[3] - x[2] + 2.5,
                                         x[4] - x[3] + 2.5,
                                         x[5] - x[4] + 2.5,
                                         x[6] - x[5] + 2.5,
                                         x[7] - x[6] + 2.5,
                                         x[8] - x[7] + 2.5,
                                         x[9] - x[8] + 2.5,
                                         x[10] - x[9] + 2.5,
                                         1.5 - x[1] + x[0],
                                         1.5 - x[2] + x[1],
                                         1.5 - x[3] + x[2],
                                         1.5 - x[4] + x[3],
                                         1.5 - x[5] + x[4],
                                         1.5 - x[6] + x[5],
                                         1.5 - x[7] + x[6],
                                         1.5 - x[8] + x[7],
                                         1.5 - x[9] + x[8],
                                         1.5 - x[10] + x[9],
                                         1.05 - sum((x[1:] + x[:-1])/2) / s_ori,
                                         sum((x[1:] + x[:-1])/2) / s_ori - 0.95]),
             'jac' : lambda x: np.array([[-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1],
                                        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1],
                                        [-x / s_ori for x in [0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5]],
                                        [x / s_ori for x in [0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.5]]])}

eq_cons = {'type': 'eq',
            'fun': lambda x: np.array([x[0] - c]),
            'jac': lambda x: np.array([1,0,0,0,0,0,0,0,0,0,0])}
     
res_rec = []
# define original variable
for i in range(len(v_test)):
    s_ori = distance[i]
    x0 = np.array(v_test[i]) 
    c = x0[0]
    res = minimize(energy, x0, method='SLSQP', jac=energy_der,
                constraints=[ineq_cons, eq_cons], options={'ftol': 1e-2, 'disp': True},
                bounds=bounds)
    res_rec.append(res)
#     # print(res.x)
with open('res.pickle', 'wb') as f:
 	pickle.dump([res_rec], f)
# plt.plot(x0, label='ori')
# plt.plot(res.x, label='opt')
# plt.legend()
# plt.show()
# print(res)

# s_ori = distance[11] 
# x0 = np.array(v_test[11]) 
# c = x0[0]
# res = minimize(energy, x0, method='SLSQP', jac=energy_der,
#             constraints=[ineq_cons, eq_cons], options={'ftol': 1e-2, 'disp': True},
#             bounds=bounds)
# print(res)

