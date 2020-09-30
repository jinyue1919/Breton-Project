import math
import numpy as np
from sqp_optimal_speed_track import energy_cal
from cluster_and_show_results import distance_cal
import pickle
from util_energy_opt import *
import copy

###### 这6个函数用来生成不同原始车速下的可行车速轨迹 ######
def higher_v_acc_acc(vel_current, vel_pred, v_last, acc=1, dec=-1):
    # vel_pred = np.insert(vel_pred, 0, vel_current, axis=0)
    t_acc_min = 0
    t_acc_max = math.floor((v_last - vel_pred[0]) / acc)
    vel_opt_rec =[] 

    for t_acc in range(t_acc_min, t_acc_max + 1):
        v_max = vel_pred[0] + t_acc * acc
        for t_acc_latter in range(0, len(vel_pred) - t_acc):
            if v_last - t_acc_latter * acc <= v_max:
                break
        vel_opt_tem1 = [vel_pred[0] + acc * t for t in range(0, t_acc)] 
        vel_opt_tem2 = [v_max] * (len(vel_pred) - t_acc - t_acc_latter)
        vel_opt_tem3 = [v_last - acc * t for t in range(0, t_acc_latter)][::-1]
        vel_opt = vel_opt_tem1 + vel_opt_tem2 + vel_opt_tem3
        vel_opt = np.array(vel_opt)
        vel_opt_rec.append(vel_opt)
    return vel_opt_rec

def higher_v_acc_dec(vel_current, vel_pred, v_last,acc=1,dec=-1):
    # vel_pred = np.insert(vel_pred, 0, vel_current, axis=0)
    t_acc_min = math.ceil((v_last - vel_pred[0]) / acc)     
    t_acc_max = min(math.floor((v_last - vel_pred[0] - dec * (len(vel_pred)-1)) / (acc - dec)) , len(vel_pred) - 1) 
    vel_opt_rec =[]
   
    for t_acc in range(t_acc_min, t_acc_max + 1):
        v_max = vel_pred[0] + t_acc * acc
        for t_dec_latter in range(0, len(vel_pred) - t_acc):
            if v_last - t_dec_latter * dec >= v_max:
                break
        vel_opt_tem1 = [vel_pred[0] + acc * t for t in range(0, t_acc)] 
        vel_opt_tem2 = [v_max] * (len(vel_pred) - t_acc - t_dec_latter)
        vel_opt_tem3 = [v_last - dec * t for t in range(0, t_dec_latter)][::-1]
        vel_opt = vel_opt_tem1 + vel_opt_tem2 + vel_opt_tem3
        vel_opt = np.array(vel_opt)
        vel_opt_rec.append(vel_opt)
    return vel_opt_rec

def higher_v_dec_acc(vel_current, vel_pred, v_last,acc=1,dec=-1):
    # vel_pred = np.insert(vel_pred, 0, vel_current, axis=0)
    t_dec_min = 0 
    t_dec_max = math.floor((v_last - vel_pred[0] - acc * (len(vel_pred) - 1)) / (dec - acc)) 
    vel_opt_rec =[]
    for t_dec in range(t_dec_min, t_dec_max + 1):
        v_min = vel_pred[0] + t_dec * dec
        for t_acc_latter in range(0, len(vel_pred) - t_dec):
            if v_last - t_acc_latter * acc <= v_min:
                break
        vel_opt_tem1 = [vel_pred[0] + dec * t for t in range(0, t_dec)] 
        vel_opt_tem2 = [v_min] * (len(vel_pred) - t_dec - t_acc_latter)
        vel_opt_tem3 = [v_last - acc * t for t in range(0, t_acc_latter)][::-1]
        vel_opt = vel_opt_tem1 + vel_opt_tem2 + vel_opt_tem3
        vel_opt = np.array(vel_opt)
        vel_opt_rec.append(vel_opt)
    return vel_opt_rec


def lower_v_acc_dec(vel_current, vel_pred, v_last, acc=1, dec=-1):
    # vel_pred = np.insert(vel_pred, 0, vel_current, axis=0)
    t_acc_min = 0
    t_acc_max = math.floor((v_last - vel_pred[0] - dec * (len(vel_pred) - 1)) / (acc - dec)) 
    vel_opt_rec = []

    for t_acc in range(t_acc_min, t_acc_max + 1):
        v_max = vel_pred[0] + t_acc * acc
        for t_dec_latter in range(0, len(vel_pred) - t_acc):
            if v_last - t_dec_latter * dec >= v_max:
                break
        vel_opt_tem1 = [vel_pred[0] + acc * t for t in range(0, t_acc)] 
        vel_opt_tem2 = [v_max] * (len(vel_pred) - t_acc - t_dec_latter)
        vel_opt_tem3 = [v_last - dec * t for t in range(0, t_dec_latter)][::-1]
        vel_opt = vel_opt_tem1 + vel_opt_tem2 + vel_opt_tem3
        vel_opt = np.array(vel_opt)
        vel_opt_rec.append(vel_opt)
    return vel_opt_rec

def lower_v_dec_acc(vel_current, vel_pred, v_last, acc=1, dec=-1):
    # vel_pred = np.insert(vel_pred, 0, vel_current, axis=0)
    t_dec_min = math.ceil((v_last - vel_pred[0]) / dec) 
    t_dec_max = math.floor((v_last - vel_pred[0] - acc * (len(vel_pred) - 1)) / (dec - acc)) 
    vel_opt_rec = []

    for t_dec in range(t_dec_min, t_dec_max + 1):
        v_min = vel_pred[0] + t_dec * dec
        for t_acc_latter in range(0, len(vel_pred) - t_dec):
            if v_last - t_acc_latter * acc <= v_min:
                break
        vel_opt_tem1 = [vel_pred[0] + dec * t for t in range(0, t_dec)] 
        vel_opt_tem2 = [v_min] * (len(vel_pred) - t_dec - t_acc_latter)
        vel_opt_tem3 = [v_last - acc * t for t in range(0, t_acc_latter)][::-1]
        vel_opt = vel_opt_tem1 + vel_opt_tem2 + vel_opt_tem3
        vel_opt = np.array(vel_opt)
        vel_opt_rec.append(vel_opt)
    return vel_opt_rec
    
def lower_v_dec_dec(vel_current, vel_pred, v_last, acc=1, dec=-1):
    # vel_pred = np.insert(vel_pred, 0, vel_current, axis=0)
    t_dec_min = 0
    t_dec_max = math.floor((v_last - vel_pred[0]) / dec) 
    vel_opt_rec = []

    for t_dec in range(t_dec_min, t_dec_max + 1):
        v_min = vel_pred[0] + t_dec * dec
        for t_dec_latter in range(0, len(vel_pred) - t_dec):
            if v_last - t_dec_latter * dec >= v_min:
                break
        vel_opt_tem1 = [vel_pred[0] + dec * t for t in range(0, t_dec)] 
        vel_opt_tem2 = [v_min] * (len(vel_pred) - t_dec - t_dec_latter)
        vel_opt_tem3 = [v_last - dec * t for t in range(0, t_dec_latter)][::-1]
        vel_opt = vel_opt_tem1 + vel_opt_tem2 + vel_opt_tem3
        vel_opt = np.array(vel_opt)
        vel_opt_rec.append(vel_opt)
    return vel_opt_rec
# vel_pred = np.linspace(10,5,num=10)
# vel_current = 10.5
# v_last = 5
# vel_opt = lower_v_dec_acc(vel_current, vel_pred, v_last)
def choose_gear(vel_current, vel_next, gear_dmd, Tm_dmd):
    length = len(gears_in_use)
    gear_tmp = np.zeros(length)
    power_tmp = np.zeros(length)
    torque_tmp = np.zeros(length)
    motor_eff_tmp = np.zeros(length)
    flag = 0
    for count, gear in enumerate(gears_in_use) :
        gear_tmp[count] = gear
        motor_speed, torque, _ = torque_calc(vel_current, vel_next, gear)
        
        if motor_speed > min(transmission_speed_max, motor_pos_speeds.max()):
            power_tmp[count] = 10000000
            torque_tmp[count] = Tm_dmd
            motor_eff_tmp[count] = -1
        else:
            Tm_max = min(interpolate_pos_torque(motor_speed), transmission_torque_max)
            Tm_min = max(interpolate_neg_torque(motor_speed), -transmission_torque_max)
            if torque > 0 and torque < Tm_max:
                torque_tmp[count] = torque
                motor_eff = interpolate_pos_motor_eff(motor_speed, torque)
                power_tmp[count] = torque * motor_speed / motor_eff
                motor_eff_tmp[count] = motor_eff
            elif torque < 0 and torque > Tm_min:
                torque_tmp[count] = torque * Reg_rate
                motor_eff = interpolate_neg_motor_eff(motor_speed, torque)
                power_tmp[count] = torque * motor_speed * Reg_rate * motor_eff
                motor_eff_tmp[count] = motor_eff
            else:
                torque_tmp[count] = min(Tm_max, max(torque, Tm_min))
                power_tmp[count] = 10000000
                motor_eff_tmp[count] = -1

    idx_min = np.argmin(power_tmp)

    if power_tmp.min() == 10000000:
        gear = gear_dmd
    else:
        gear = gear_tmp[idx_min]  
        flag = 1
    return [gear], flag 

#### 这个函数用来计算这种优化方法的结果，包括flag、优化车速、优化挡位、优化转矩、电机效率 #### 
########### 这部分用来计算能耗最优的速度轨迹 ########### 
def energy_opt_v3(gears_in_use, gear_pre, vel_current, vel_pred, Tm_dmd, gear_dmd, acc=1, dec=-1, gear_pre_duration=3):
    vel_pred = np.insert(vel_pred, 0, vel_current, axis=0)
    v_last_range = np.linspace(vel_pred[-1] * 0.95, vel_pred[-1] * 1.05, num = 5)                   # vel_pred:m/s
    distance_ori = distance_cal(vel_pred)
    energy_ori = energy_cal(vel_pred)

    vel_opt_rec = []
    energy_opt_generate_v_rec = []

    for v_last in v_last_range:
        if v_last < vel_pred[0]:
            v1 = lower_v_acc_dec(vel_current, vel_pred, v_last)
            v2 = lower_v_dec_acc(vel_current, vel_pred, v_last)
            v3 = lower_v_dec_dec(vel_current, vel_pred, v_last)
            vel_opt_tem = v1+v2+v3
            index = []
            for i in range(0, len(vel_opt_tem)):
                distane_opt_generate_v_tem = distance_cal(vel_opt_tem[i]) 
                energy_opt_generate_v_tem = energy_cal(vel_opt_tem[i])
                if abs(distane_opt_generate_v_tem / distance_ori - 1) <= 0.1 and energy_opt_generate_v_tem < energy_ori:
                    index.append(i)
            if index == []:
                vel_opt = vel_pred
                energy_opt_generate_v = energy_cal(vel_opt)
            else:
                min_energy_position = [energy_cal(vel_opt_tem[i]) for i in index].index(min([energy_cal(vel_opt_tem[i]) for i in index]))
                energy_opt_generate_v = min([energy_cal(vel_opt_tem[i]) for i in index])
                vel_opt = vel_opt_tem[index[min_energy_position]] 
        else:
            v1 = higher_v_acc_acc(vel_current, vel_pred, v_last)
            v2 = higher_v_acc_dec(vel_current, vel_pred, v_last)
            v3 = higher_v_dec_acc(vel_current, vel_pred, v_last)
            vel_opt_tem = v1+v2+v3
            index = []
            for i in range(0, len(vel_opt_tem)):
                distane_opt_generate_v_tem = distance_cal(vel_opt_tem[i]) 
                energy_opt_generate_v_tem = energy_cal(vel_opt_tem[i])
                if abs(distane_opt_generate_v_tem / distance_ori - 1) <= 0.1 and energy_opt_generate_v_tem < energy_ori:
                    index.append(i)
            if index == []:
                vel_opt = vel_pred
                energy_opt_generate_v = energy_cal(vel_opt)
            else:
                min_energy_position = [energy_cal(vel_opt_tem[i]) for i in index].index(min([energy_cal(vel_opt_tem[i]) for i in index]))
                energy_opt_generate_v = min([energy_cal(vel_opt_tem[i]) for i in index])
                vel_opt = vel_opt_tem[index[min_energy_position]] 
        vel_opt_rec.append(vel_opt)
        energy_opt_generate_v_rec.append(energy_opt_generate_v)
    min_energy_pos_v_last_range = energy_opt_generate_v_rec.index(min(energy_opt_generate_v_rec))
    vel_opt = vel_opt_rec[min_energy_pos_v_last_range]
 ################################# 选档  #################################   
    vel_next = vel_opt[1]
    gear_opt, flag = choose_gear(vel_current, vel_next, gear_dmd, Tm_dmd)
    return vel_opt, gear_opt, flag

