#!/usr/bin/env python3
"""Subscribes to /scan (lidar) and /vrpn_client_node/jackal/pose (position & orientation) — ROS1."""

import numpy as np
import math
from visualize_lidar_scan import generate_occupancy_grid, visualize_occupancy_grid_polar
from cell_config import cell, cell_ls
from find_controller_orientation import control_gain_load
import os
import matplotlib.pyplot as plt
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, Twist


def quaternion_to_yaw(q):
    """Extract yaw (heading) in radians from a quaternion (x, y, z, w)."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def generate_occupancy_grid_polar(scan_data, num_angle_bins=120, num_range_bins=40):
    """
    Generate an occupancy grid in polar coordinates based on lidar scan data.
    
    Args:
        scan_data: scan data from lidar
        num_angle_bins: Number of angle bins (default: 120, corresponding to 3-degree increments)
        num_range_bins: Number of range bins (default: 40)
    
    Returns:
        occupancy_grid: 2D numpy array of shape (num_range_bins, num_angle_bins) with values:
            - 1.0: occupied cell (contains obstacles)
            - 0.0: free cell (along ray path, no obstacles)
        polar_params: Dictionary with 'angle_min', 'angle_max', 'angle_increment', 
                     'range_min', 'range_max', 'range_increment' defining the grid parameters
    """
    # Load data
    ranges = np.array(scan_data.ranges)
    angle_min = scan_data.angle_min
    angle_max = scan_data.angle_max
    angle_increment = scan_data.angle_increment
    range_min = scan_data.range_min
    range_max = scan_data.range_max
    
    # Generate angles for lidar data
    num_points = len(ranges)
    if angle_increment > 0:
        angles = np.arange(angle_min, angle_min + num_points * angle_increment, angle_increment)
        angles = angles[:num_points]
    else:
        angles = np.linspace(angle_min, angle_max, num_points)
    
    # Filter out invalid ranges
    valid_mask = np.isfinite(ranges) & (ranges >= range_min) & (ranges <= range_max)
    ranges_valid = ranges[valid_mask]
    angles_valid = angles[valid_mask]
    
    # Define polar grid parameters
    # Angle bins: 120 bins with 3-degree increments covering full 360 degrees
    angle_bin_increment = 3.0 * np.pi / 180.0  # 3 degrees in radians
    angle_grid_min = 0.0  # Start from 0 radians (East)
    angle_grid_max = num_angle_bins * angle_bin_increment  # Full circle
    
    # Range bins: 40 bins from range_min to range_max
    # range_increment = (range_max - range_min) / num_range_bins
    range_increment = (5 - 0) / num_range_bins
    
    # Initialize occupancy grid (0.5 = unknown)
    occupancy_grid = np.full((num_range_bins, num_angle_bins), 0.0, dtype=np.float32)
    
    # Function to convert (angle, range) to grid indices
    def polar_to_grid(angle, range_val):
        """Convert polar coordinates to grid indices."""
        # Normalize angle to [0, 2π) range
        angle_norm = angle % (2 * np.pi)
        if angle_norm < 0:
            angle_norm += 2 * np.pi
        
        # Find angle bin index
        angle_bin = int(angle_norm / angle_bin_increment)
        angle_bin = np.clip(angle_bin, 0, num_angle_bins - 1)
        
        # Find range bin index
        range_bin = int((range_val - range_min) / range_increment)
        range_bin = np.clip(range_bin, 0, num_range_bins - 1)
        
        return range_bin, angle_bin
    
    # Mark occupied cells (cells containing obstacle endpoints)
    for angle, range_val in zip(angles_valid, ranges_valid):
        range_bin, angle_bin = polar_to_grid(angle, range_val)
        occupancy_grid[range_bin, angle_bin] = 1.0  # Occupied
    
    # Mark free cells (cells along ray path from robot to obstacle)
    for angle, range_val in zip(angles_valid, ranges_valid):
        if range_val > 0:
            # Get grid position of obstacle
            end_range_bin, end_angle_bin = polar_to_grid(angle, range_val)
            
            # Mark all range bins from 0 to obstacle as free
            # (except the obstacle cell itself which is already marked as occupied)
            for r_bin in range(end_range_bin):
                # Keep obstacle cells as occupied, mark others as free
                if occupancy_grid[r_bin, end_angle_bin] != 1.0:
                    occupancy_grid[r_bin, end_angle_bin] = 0.0  # Free
    
    # Store polar parameters for reference
    polar_params = {
        'angle_min': angle_grid_min,
        'angle_max': angle_grid_max,
        'angle_increment': angle_bin_increment,
        'num_angle_bins': num_angle_bins,
        'range_min': range_min,
        'range_max': range_max,
        'range_increment': range_increment,
        'num_range_bins': num_range_bins
    }
    # fig, ax = visualize_occupancy_grid_polar(occupancy_grid, polar_params)
    # fig.savefig('occupancy_grid_polar.png')
    # print('occupancy_grid_polar.png saved')
    # plt.close(fig)
    return occupancy_grid, polar_params

def load_RSC_data():
    dir = 'allocentric_ratemaps/RSC/data'
    files = os.listdir(dir)
    output = []
    for file in files:
        data = np.load(os.path.join(dir, file))
        output.append(data)
    return np.array(output)


class ScanPoseSubscriber(object):
    """Subscribes to /scan and /vrpn_client_node/jackal/pose (ROS1)."""

    def __init__(self):
        self._latest_scan = None
        self._latest_pose = None
        self._position = (0.0, 0.0, 0.0)
        self._orientation_yaw = 0.0
       
        self._scan_sub = rospy.Subscriber(
            "/scan", LaserScan, self._scan_cb, queue_size=10
        )
        self._pose_sub = rospy.Subscriber(
            "/vrpn_client_node/jackal/pose", PoseStamped, self._pose_cb, queue_size=10
        )
        self._control_linear_pub = rospy.Publisher(
            "/controller_output_linear", Twist, queue_size=10
        )
        self._control_unicycle_pub = rospy.Publisher(
            "/cmd_vel", Twist, queue_size=10
        )
        rospy.loginfo("Subscribed to /scan and /vrpn_client_node/jackal/pose")
        rospy.loginfo("Publishing controller output to /controller_output")
        self.cell_ls = cell_ls
        self.control_gain_load = control_gain_load()
        self.RSC_data = load_RSC_data()
        self.current_grid_occ = None
        self.current_cell = None
        self.current_K = None
        self.current_Kb = None
        self.current_measurement = None
        self.current_grid_occ = None

        self._traj_capacity = 500
        self._traj_idx = 0
        self.pos_ls = [None] * self._traj_capacity
        self.control_ls = [None] * self._traj_capacity
        self.K_ls = [None] * self._traj_capacity
        self.Kb_ls = [None] * self._traj_capacity
        self.measurement_ls = [None] * self._traj_capacity
        self.grid_occ_ls = [None] * self._traj_capacity

    def _find_cell(self, position):
        """Find which cell the robot is currently in"""
        # try:
        ls_flag = []
        for i in range(len(self.cell_ls)):
            ls_flag.append(self.cell_ls[i].check_in_polygon(np.reshape(position, (1, 2))))
        
        cell_indices = [i for i, x in enumerate(ls_flag) if x]
        if len(cell_indices) == 0:
            return None
        # if 1.5<position[0]<2.0 and -0.1 < position[1] <0.1:
        #     return 1
        return cell_indices[0]

    def controller(self):
        if self.current_cell is None:
            return np.zeros((2,1))
        #check if the current_grid_occ exits
        if self.current_grid_occ is None:
            return np.zeros((2,1))
        K, Kb = self.control_gain_load.interpolate_contorlgains(self.current_cell, self._orientation_degree)       
        measurement = []
        for i in range(len(self.RSC_data)):
            measurement.append(np.sum(self.RSC_data[i]*self.current_grid_occ.T))

        measurement = np.array(measurement)
        measurement = measurement.reshape(-1, 1)
        u = K[0]@measurement+Kb

        self.current_K = K
        self.current_Kb = Kb
        self.current_measurement = measurement
        
        
        # Normalize and scale
        speed = 3.0
        u_normalized = u / np.linalg.norm(u)
        u_scaled = u_normalized * speed
        return u_scaled.reshape(2,1)




    def _scan_cb(self, msg):
        self.lidar_scan = msg
        self.current_grid_occ, self.polar_params = generate_occupancy_grid_polar(self.lidar_scan)
        rospy.loginfo_throttle(1.0, "received scan")
        u = self.controller()
        v, omega = self.offest_unicycle_model(u)
        self.publish_control_unicycle_model(v, omega)
        rospy.loginfo( "control : (%.3f, %.3f)" % (u[0], u[1]))
        rospy.loginfo( "vel and omeg: (%.3f, %.3f)" % (v, omega))
        rospy.loginfo( "current_cell: "+str(self.current_cell))
        rospy.loginfo("pose: x=%.3f y=%.3f yaw=%.3f deg" % (self._position[0], self._position[1], self._orientation_degree))
        # rospy.loginfo_throttle(1.0, "pose: x=%.3f y=%.3f yaw=%.3f rad" % (p.x, p.y, self._orientation_yaw))
        self.linear_controller = u
        self.publish_control(u)
        if self._traj_idx < self._traj_capacity:
            self.pos_ls[self._traj_idx] = self._position
            self.control_ls[self._traj_idx] = u
            self.K_ls[self._traj_idx] = self.current_K
            self.Kb_ls[self._traj_idx] = self.current_Kb
            self.measurement_ls[self._traj_idx] = self.current_measurement
            self.grid_occ_ls[self._traj_idx] = self.current_grid_occ
            rospy.loginfo("traj_idx: %d" % self._traj_idx)
            self._traj_idx += 1

    def save_trajectory(self):
        """Save trajectory lists to disk (call on shutdown)."""
        if self._traj_idx == 0:
            rospy.loginfo("No trajectory data to save.")
            return
        os.makedirs('trj_data', exist_ok=True)
        n = self._traj_idx
        np.save('trj_data/pos_ls.npy', self.pos_ls[:n])
        np.save('trj_data/control_ls.npy', self.control_ls[:n])
        np.save('trj_data/K_ls.npy', self.K_ls[:n])
        np.save('trj_data/Kb_ls.npy', self.Kb_ls[:n])
        np.save('trj_data/measurement_ls.npy', self.measurement_ls[:n])
        np.save('trj_data/grid_occ_ls.npy', self.grid_occ_ls[:n])
        rospy.loginfo("Saved trajectory (%d samples) to disk." % n)

    def publish_control(self, u):
        'publish the control synthesis  u = k*measurement + kb'
        twist_msg = Twist()
        twist_msg.linear.x = float(u[0])
        twist_msg.linear.y = float(u[1])
        twist_msg.linear.z = 0.0
        twist_msg.angular.x = 0.0
        twist_msg.angular.y = 0.0
        twist_msg.angular.z = 0.0
        self._control_linear_pub.publish(twist_msg)

    def clamp(self, x: float, lo: float, hi: float) -> float:
        'clamp the value to the range [lo, hi]'
        return max(lo, min(hi, x))


    def offest_unicycle_model(self, u):
        # Map to v, omega
        # epsilon is the offset of the unicycle model
        self.epsilon = 0.5
        # self.epsilon = 0.1
        J_inv = np.array([
            [np.cos(self._orientation_yaw), np.sin(self._orientation_yaw)],
            [-np.sin(self._orientation_yaw)/self.epsilon, np.cos(self._orientation_yaw)/self.epsilon]
        ])
        v_omega = np.dot(J_inv, u)
        v, omega = v_omega[0], v_omega[1]
        v = self.clamp(v, -0.5, 0.5)
        omega = self.clamp(omega, -1, 1)
        return v, omega

    def publish_control_unicycle_model(self, v, omega):
        'publish the control to the unicycle model'
        twist_msg = Twist()
        twist_msg.linear.x = float(v)
        twist_msg.linear.y = 0.0
        twist_msg.angular.z = float(omega)
        self._control_unicycle_pub.publish(twist_msg)
        

    def _pose_cb(self, msg):
        self._latest_pose = msg
        p = msg.pose.position
        q = msg.pose.orientation
        self._position = (p.x, p.y)
        self._orientation_yaw = quaternion_to_yaw(q)
        self._orientation_degree = self._orientation_yaw * 180 / np.pi % 360
        # rospy.loginfo_throttle(1.0, "pose: x=%.3f y=%.3f yaw=%.3f rad" % (p.x, p.y, self._orientation_yaw))
        self.current_cell = self._find_cell(self._position)

        
        if self.current_cell is not None:
            rospy.loginfo_throttle(1.0, "current cell: %d" % (self.current_cell))
        else:
            rospy.loginfo_throttle(1.0, "current cell: None")

    @property
    def latest_scan(self):
        return self._latest_scan

    @property
    def latest_pose(self):
        return self._latest_pose

    @property
    def position(self):
        return self._position

    @property
    def orientation_yaw(self):
        return self._orientation_yaw




def main():
    rospy.init_node("scan_pose_subscriber", anonymous=False)
    node = ScanPoseSubscriber()
    rospy.on_shutdown(node.save_trajectory)
    rospy.spin()


if __name__ == "__main__":
    main()
