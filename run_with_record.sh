#!/bin/bash

# Run rosbag record in background and controller_node in foreground.
# Recording stops when the controller exits (Ctrl+C).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Start rosbag record in background
rosbag record -o test /scan /vrpn_client_node/jackal/pose /controller_output_linear /cmd_vel &
ROSBAG_PID=$!

# Ensure rosbag is killed when this script exits (including Ctrl+C)
cleanup() {
  echo "Stopping rosbag record (PID $ROSBAG_PID)..."
  kill $ROSBAG_PID 2>/dev/null
  exit 0
}
trap cleanup EXIT INT TERM

# Run controller (blocking)
python3 controller_node.py
