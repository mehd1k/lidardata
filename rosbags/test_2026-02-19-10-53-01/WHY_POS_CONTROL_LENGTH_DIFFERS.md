# Why pos_ls and control_ls have different lengths (robot vs replay)

For the **same** rosbag, when you run:

- **On the robot**: `find_diff/controller_node_on_robot.py` (ROS1) → writes `trj_data_on_robot/` (e.g. **266** samples)
- **On replay**: `controller_node.py` (ROS2) while replaying the bag → writes `trj_data_ros2/` (e.g. **400** samples)

So the **number of entries** in `pos_ls` and `control_ls` is different between the two runs (266 vs 400). Within a single run, `len(pos_ls) == len(control_ls)` always, because both are appended together in the same callback.

---

## Why the counts differ: scan callback rate vs processing speed

In both nodes, **one pair (position, control) is appended per `/scan` callback** — in `_scan_cb` only:

- **Robot** (`find_diff/controller_node_on_robot.py`): `_scan_cb` does:

  - `generate_occupancy_grid_polar(scan)`
  - `controller()` (RSC data, interpolate_contorlgains, matrix multiply)
  - `offest_unicycle_model`, `publish_control`, then **`pos_ls.append`, `control_ls.append`**
  - **6× `np.save(...)`** (pos_ls, control_ls, K_ls, Kb_ls, measurement_ls, grid_occ_ls)
- **Replay** (`controller_node.py` ROS2): same idea — append in `_scan_cb` after computing control and saving.

So:

- **Number of samples = number of times `_scan_cb` was invoked.**

That depends on how many `/scan` messages the node **actually processes**, not only how many are in the bag.

### On the robot

- `/scan` is published at a fixed rate (e.g. 10–40 Hz).
- `_scan_cb` is **heavy** (occupancy grid, controller, 6× disk writes). If the callback takes longer than the period between two scans, the single-threaded ROS spinner cannot call `_scan_cb` for every message.
- With `queue_size=10`, when the callback is still running, new scans queue up; once the queue is full, **older messages are dropped**. So the robot may only **process a subset** of the published scans (e.g. 266 out of 400).

So: **fewer samples on the robot = dropped scan messages due to slow callback.**

### On replay (computer)

- The bag is played back; the same controller code runs on a typically **faster machine**.
- Callback runs faster (CPU, disk), so it can keep up with the replay rate.
- **Every** `/scan` message from the bag triggers `_scan_cb` → you get one sample per scan in the bag (e.g. 400).

So: **more samples on replay = no (or fewer) dropped scans.**

---

## Summary

| Run    | Samples (e.g.) | Reason                                                                      |
| ------ | -------------- | --------------------------------------------------------------------------- |
| Robot  | 266            | Heavy `_scan_cb` (grid + control + 6× save); some scan messages dropped. |
| Replay | 400            | Same callback but faster machine; every scan in the bag is processed.       |

So the length difference is **not** because position and control are logged in different places; it’s because the **number of scan callbacks** that run is different (robot processes fewer scans than are in the bag).

---

## How to reduce the difference (optional)

1. **Lighten `_scan_cb` on the robot**

   - Don’t save to disk inside the callback.
   - Use a timer (e.g. 1 Hz) to copy and save `pos_ls`, `control_ls`, etc., so the scan callback only does grid + control + append.
2. **Or accept the mismatch**

   - When comparing robot vs replay, align by **time** (using bag timestamps) or by **scan index** (if you know which scan messages were dropped on the robot).
