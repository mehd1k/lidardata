import numpy as np
import matplotlib.pyplot as plt
from cell_config import cell_ls


def find_cell(cell_ls, position):
        """Find which cell the robot is currently in"""
        # try:
        ls_flag = []
        for i in range(len(cell_ls)):
            ls_flag.append(cell_ls[i].check_in_polygon(np.reshape(position, (1, 2))))
        
        cell_indices = [i for i, x in enumerate(ls_flag) if x]
        if len(cell_indices) == 0:
            return None
        return cell_indices[0]

control_r1 = np.load('/home/mehdi/lidardata/rosbags/test_2026-02-19-10-53-01/trj_data_on_robot/control_ls.npy')
control_r2 = np.load('/home/mehdi/lidardata/rosbags/test_2026-02-19-10-53-01/trj_data_ros2/control_ls.npy')
kb_r1 = np.load('/home/mehdi/lidardata/rosbags/test_2026-02-19-10-53-01/trj_data_on_robot/Kb_ls.npy')
kb_r2 = np.load('/home/mehdi/lidardata/rosbags/test_2026-02-19-10-53-01/trj_data_ros2/Kb_ls.npy')
measurement_r1 = np.load('/home/mehdi/lidardata/rosbags/test_2026-02-19-10-53-01/trj_data_on_robot/measurement_ls.npy')
measurement_r2 = np.load('/home/mehdi/lidardata/rosbags/test_2026-02-19-10-53-01/trj_data_ros2/measurement_ls.npy')
k_r1 = np.load('/home/mehdi/lidardata/rosbags/test_2026-02-19-10-53-01/trj_data_on_robot/K_ls.npy')
k_r2 = np.load('/home/mehdi/lidardata/rosbags/test_2026-02-19-10-53-01/trj_data_ros2/K_ls.npy')
pos_r1 = np.load('/home/mehdi/lidardata/rosbags/test_2026-02-19-10-53-01/trj_data_on_robot/pos_ls.npy')
pos_r2 = np.load('/home/mehdi/lidardata/rosbags/test_2026-02-19-10-53-01/trj_data_ros2/pos_ls.npy')

cell_indices_r1 = []
cell_indices_r2 = []
for i in range(len(pos_r1)):
    cell_indices_r1.append(find_cell(cell_ls, pos_r1[i]))
for i in range(len(pos_r2)):
    cell_indices_r2.append(find_cell(cell_ls, pos_r2[i]))


print(kb_r1.shape)
print(kb_r2.shape)
print(control_r1.shape)
print(control_r2.shape)
print(measurement_r1.shape)
print(measurement_r2.shape)
print(k_r1.shape)
print(k_r2.shape)
fig, ax = plt.subplots(6,1)
ax[0].plot(control_r1[:,0],label='control_r1')
ax[0].plot(control_r2[:,0],label='control_r2')
ax[1].plot(kb_r1[:,0],label='kb_r1')
ax[1].plot(kb_r2[:,0],label='kb_r2')
ax[2].plot(kb_r1[:,1],label='kb_r1')
ax[2].plot(kb_r2[:,1],label='kb_r2')
ax[3].plot(np.mean(measurement_r1,axis=1),label='measurement_r1')
ax[3].plot(np.mean(measurement_r2,axis=1),label='measurement_r2')
ax[4].plot(np.mean(k_r1,axis=3)[:,0,0],label='k_r1')
ax[4].plot(np.mean(k_r2,axis=3)[:,0,0],label='k_r2')
ax[5].plot(cell_indices_r1,label='cell_indices_r1')
ax[5].plot(cell_indices_r2,label='cell_indices_r2')
plt.legend()
plt.show()