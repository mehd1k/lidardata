#!/usr/bin/env python2
"""
Extract control, pose, and scan data from a ROS1 bag file.
For each /controller_output_linear message, finds the closest /vrpn_client_node/jackal/pose
and /scan by timestamp and saves them.

Usage: python2 extract_bag_data.py <bag_path> [output_dir]
Output: pos_ls.npy, control_ls.npy, scan_ls.npy in output_dir
"""
import sys
import os
import numpy as np
import rosbag

def find_closest(ts_ns, timestamps_ns):
    """Find index of timestamp in timestamps_ns closest to ts_ns."""
    idx = np.searchsorted(timestamps_ns, ts_ns)
    if idx == 0:
        return 0
    if idx >= len(timestamps_ns):
        return len(timestamps_ns) - 1
    if abs(timestamps_ns[idx] - ts_ns) < abs(timestamps_ns[idx-1] - ts_ns):
        return idx
    return idx - 1

def main():
    if len(sys.argv) < 2:
        print("Usage: python2 extract_bag_data.py <bag_path> [output_dir]")
        sys.exit(1)
    
    bag_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(bag_path))
    
    if not os.path.exists(bag_path):
        print("Error: Bag file not found:", bag_path)
        sys.exit(1)
    
    control_topic = '/controller_output_linear'
    pose_topic = '/vrpn_client_node/jackal/pose'
    scan_topic = '/scan'
    
    # First pass: collect all messages with timestamps
    pose_data = []  # (timestamp_ns, x, y)
    control_data = []  # (timestamp_ns, ux, uy)
    scan_data = []  # (timestamp_ns, ranges_array)
    
    bag = rosbag.Bag(bag_path)
    
    for topic, msg, t in bag.read_messages(topics=[pose_topic, control_topic, scan_topic]):
        ts_ns = t.to_nsec()
        if topic == pose_topic:
            x = msg.pose.position.x
            y = msg.pose.position.y
            pose_data.append((ts_ns, x, y))
        elif topic == control_topic:
            ux = msg.linear.x
            uy = msg.linear.y
            control_data.append((ts_ns, ux, uy))
        elif topic == scan_topic:
            ranges = np.array(msg.ranges, dtype=np.float32)
            scan_data.append((ts_ns, ranges))
    
    bag.close()
    
    if len(control_data) == 0:
        print("Error: No control messages found in bag")
        sys.exit(1)
    
    # Build sorted arrays for lookup
    pose_ts = np.array([p[0] for p in pose_data])
    control_ts = np.array([c[0] for c in control_data])
    scan_ts = np.array([s[0] for s in scan_data])
    
    # For each control, find closest pose and scan
    pos_ls = []
    control_ls = []
    scan_ls = []
    
    for i, (ctrl_ts, ux, uy) in enumerate(control_data):
        pose_idx = find_closest(ctrl_ts, pose_ts)
        scan_idx = find_closest(ctrl_ts, scan_ts)
        
        pos_ls.append([pose_data[pose_idx][1], pose_data[pose_idx][2]])
        control_ls.append([ux, uy])
        scan_ls.append(scan_data[scan_idx][1])
    
    pos_ls = np.array(pos_ls)
    control_ls = np.array(control_ls)
    scan_ls = np.array(scan_ls)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(os.path.join(output_dir, 'pos_ls.npy'), pos_ls)
    np.save(os.path.join(output_dir, 'control_ls.npy'), control_ls)
    np.save(os.path.join(output_dir, 'scan_ls.npy'), scan_ls)
    
    print("Extracted {} control/pose/scan tuples".format(len(pos_ls)))
    print("Saved to:", output_dir)

if __name__ == '__main__':
    main()
