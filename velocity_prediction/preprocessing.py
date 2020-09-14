#%%
import os
import csv
import matplotlib.pyplot as plt
import numpy as np


# SD_CARDS = ['0614-0712-09299',
#            '0617-0712-00755',
#            '0620-0712-00788',
#            '0621-0712-05266',
#            '0621-0712-08758',
#            '0629-0712-00767',
#            '0702-0712-00757']

def FindFileLost(FilePath1, FilePath2):
    SD_CARDS1 = os.listdir(FilePath1)
    SD_CARDS2 = os.listdir(FilePath2)
    LostFiles = []
    for num, SD_CARD1 in enumerate(SD_CARDS1):
        Files2BeChecked = os.listdir(os.path.join(FilePath1, SD_CARD1))
        for file1 in Files2BeChecked:
            if 'iwdz' in file1:
                flag = 0
                for file2 in os.listdir(os.path.join(FilePath2, SD_CARDS2[num])):
                    if file2[:-4] == file1[:-5]:
                        flag += 1
                if flag == 0 and 'iwdz' in file1:
                    # print(file1)
                    LostFiles.append(f'{SD_CARD1}: {file1}')
    return LostFiles

# Data is collected for 2019-0614 to 2019-0712

# Lists to store CAN data
CANData_vel = []
CANData_gear = []

file_vel = np.array([])
file_gear = np.array([])
file_final = np.array([])
REAL_NUM = 0
# PATH = '/Users/tujiayu/Dev/BretonProject/Data/'
PATH = '/Volumes/David/BretonProject-data/Data'
SD_CARDS = os.listdir(PATH)
SD_CARDS.sort()

#%%
for SD_num, SD_CARD in enumerate(SD_CARDS):
    
    path2SD_CARD = os.path.join(PATH, SD_CARD)
    Files2BeConverted = os.listdir(path2SD_CARD)
    Files2BeConverted.sort()
    DATE_INIT = Files2BeConverted[0][4:8]  # Date
    file_final = np.array([])  # Stores data of one day

    for file in Files2BeConverted:
        
        # Change of date
        if DATE_INIT != file[4:8]:
            # file_final = file_final.tolist()
            # file_final.insert(0, ['velocity', 'accelerator', 'brake', 'gear'])
            if file_final.size == 0:
                print(f'Data in SD card {SD_CARD}, date {DATE_INIT} is empty.')
            else:
                # with open(f'{DATE_INIT}_{SD_num}.csv', 'w', newline='') as f:
                #     writer = csv.writer(f)
                #     writer.writerows(file_final)
                np.savetxt(f'{DATE_INIT}_{SD_num}.csv', file_final, fmt='%.4f')
                print(f'Data in SD card {SD_CARD}, date {DATE_INIT} has been converted successfuly!')
            
            # Reset date
            DATE_INIT = file[4:8]
            file_final = np.array([])
        
        # Path to a single file
        path_final = os.path.join(path2SD_CARD, file)
        
        # Explicitly specify encoding to avoid UnicodeDecideError
        with open(path_final, encoding="ISO-8859-1") as f:
            for line in f:
                if '18ff44a8x' in line:
                    CANData_vel.append(line)
                elif '18ff48a8x' in line:
                    CANData_gear.append(line)
                
        for count, line in enumerate(CANData_vel):
            frame = line.split()
            speed_frame = (int(frame[6][1], 16) + int(frame[6][0], 16) * 16 + int(frame[7][1], 16) * 16 ** 2 +
                int(frame[7][0], 16) * 16 ** 3) * 0.00390625
            acc_frame = (int(frame[9][1], 16) + int(frame[9][0], 16) * 16) * 0.005
            brake_frame = (int(frame[10][1], 16) + int(frame[10][0], 16) * 16) * 0.005
            
            if count == 0:
                file_vel = np.array([speed_frame, acc_frame, brake_frame])
                # file_vel = np.array([speed_frame])
            else:
                file_vel = np.vstack((file_vel, [speed_frame, acc_frame, brake_frame]))
                # file_vel = np.vstack((file_vel, [speed_frame]))
                
        for count, line in enumerate(CANData_gear):
            frame = line.split()
            if len(frame) == 14:
                if frame[6] == '0F':
                    gear_frame = None
                else:
                    gear_frame = int(frame[6][1], 16) - 1
                    gear_flag = int(frame[12][1], 16)
                if count == 0:
                    file_gear = np.array([gear_frame, gear_flag])
                else:
                    file_gear = np.vstack((file_gear, [gear_frame, gear_flag]))
            else:
                print('Abnormal frame detected!')
            
        len_min = min(file_gear.shape[0], file_vel.shape[0])
        if file_gear.shape[0] > len_min:
            print('Different size: file_gear size: ', file_gear.shape[0], '; file_vel size: ', file_vel.shape[0])
            file_gear = np.delete(file_gear, [i for i in range(len_min, file_gear.shape[0])], axis=0)
        elif file_vel.shape[0] > len_min:
            print('Different size: file_gear size: ', file_gear.shape[0], '; file_vel size: ', file_vel.shape[0])
            file_vel = np.delete(file_vel, [i for i in range(len_min, file_vel.shape[0])], axis=0)
        else:
            print('Same size')

        per_file = np.hstack((file_vel, file_gear))
        # per_file = file_vel
        
        # When processing the first file in a date, file_final is empty, cannot do vstack directly
        if file_final.shape[0] == 0:
            file_final = per_file
        elif per_file.size != 0:
            file_final = np.vstack((file_final, per_file))
            
        CANData_vel.clear()
        CANData_gear.clear()
        file_vel = np.array([])
        file_gear = np.array([])
    # break

# vel = []; acc = []; brake = []
# with open(mu'{DATE_INIT}_{REAL_NUM}_test.csv') as csv_file:
#     csv_reader = csv.reader(csv_file, delimiter=',')
#     for count, line in enumerate(csv_reader):
#         vel.append(line[0])
#         acc.append(line[1])
#         brake.append(line[2])
#
# del vel[0], acc[0], brake[0]
#
# vel = [float(i) for i in vel]
# acc = [float(i) for i in acc]
# brake = [float(i) for i in brake]
#
# plt.plot([num for num, ele in enumerate(vel)], vel)
# plt.plot([num for num, ele in enumerate(acc)], acc)
# plt.plot([num for num, ele in enumerate(brake)], brake)
# plt.legend(['velocity', 'acc', 'brake'], loc='upper right')
# plt.show()