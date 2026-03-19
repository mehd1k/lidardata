"""
Extract pose data (x, y, yaw) from a ROS 2 bag (SQLite .db3) containing PoseStamped messages.
"""

import math
from pathlib import Path
from typing import Union

import numpy as np

# Default pose topic in data/data.db3 (VRPN / Jackal)
DEFAULT_POSE_TOPIC = "/vrpn_client_node/jackal/pose"


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Extract yaw (heading) in radians from a quaternion (x, y, z, w)."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def extract_poses(
    db_path: Union[str, Path] = "data/data.db3",
    topic: Union[str, None] = None,
    output_path: Union[str, Path, None] = None,
) -> np.ndarray:
    """
    Extract pose data (x, y, yaw) from a ROS 2 bag database (data.db3).

    Reads geometry_msgs/msg/PoseStamped from the given topic, extracts position (x, y)
    and yaw from the quaternion orientation. Returns and optionally saves as a numpy
    array of shape (N, 3) with dtype float64.

    Parameters
    ----------
    db_path : str or Path
        Path to the bag directory or to the .db3 file (e.g. "data/data.db3").
    topic : str, optional
        Pose topic name. Defaults to "/vrpn_client_node/jackal/pose".
    output_path : str or Path, optional
        If set, save the array to this path with np.save() (e.g. "data/poses.npy").

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) with columns [x, y, yaw]. yaw is in radians.

    Raises
    ------
    FileNotFoundError
        If db_path does not exist.
    ValueError
        If the topic is not found or has no PoseStamped messages.
    """
    topic = topic or DEFAULT_POSE_TOPIC
    db_path = Path(db_path)
    if db_path.is_file() and db_path.suffix == ".db3":
        bag_dir = db_path.parent
        db3_path = db_path
    elif db_path.is_dir():
        bag_dir = db_path
        db3_files = list(db_path.glob("*.db3"))
        if not db3_files:
            raise FileNotFoundError(f"No .db3 file found in directory: {db_path}")
        db3_path = db3_files[0]
    else:
        raise FileNotFoundError(f"Bag path does not exist: {db_path}")

    if not bag_dir.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_dir}")

    # Try rosbag2_py first
    try:
        poses = _extract_with_rosbag2_py(bag_dir=bag_dir, topic=topic)
    except ImportError:
        poses = _extract_with_sqlite(db3_path=db3_path, topic=topic)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, poses)

    return poses


def _extract_with_sqlite(db3_path: Path, topic: str) -> np.ndarray:
    import sqlite3

    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    PoseStamped = get_message("geometry_msgs/msg/PoseStamped")
    conn = sqlite3.connect(str(db3_path))
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM topics WHERE name = ?", (topic,))
    row = cursor.fetchone()
    if row is None:
        conn.close()
        raise ValueError(f"Topic not found: {topic}")

    topic_id = row[0]
    cursor.execute(
        "SELECT id, timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
        (topic_id,),
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        raise ValueError(f"No messages on topic: {topic}")

    poses_list = []
    for msg_id, timestamp, data in rows:
        try:
            msg = deserialize_message(data, PoseStamped)
        except Exception as e:
            print(f"Warning: skip message id={msg_id} timestamp={timestamp}: {e}")
            continue
        p = msg.pose.position
        o = msg.pose.orientation
        yaw = quaternion_to_yaw(o.x, o.y, o.z, o.w)
        poses_list.append([p.x, p.y, yaw])

    return np.array(poses_list, dtype=np.float64)


def _extract_with_rosbag2_py(bag_dir: Path, topic: str) -> np.ndarray:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    storage_options = StorageOptions()
    storage_options.uri = str(bag_dir)
    storage_options.storage_id = "sqlite3"
    converter_options = ConverterOptions()
    converter_options.input_serialization_format = "cdr"
    converter_options.output_serialization_format = "cdr"

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    if not any(t.name == topic for t in topic_types):
        raise ValueError(f"Topic not found: {topic}. Available: {[t.name for t in topic_types]}")

    reader.set_read_filter([topic])
    PoseStamped = get_message("geometry_msgs/msg/PoseStamped")
    poses_list = []
    while reader.has_next():
        read_topic, data, _ = reader.read_next()
        if read_topic != topic:
            continue
        msg = deserialize_message(data, PoseStamped)
        p = msg.pose.position
        o = msg.pose.orientation
        yaw = quaternion_to_yaw(o.x, o.y, o.z, o.w)
        poses_list.append([p.x, p.y, yaw])

    return np.array(poses_list, dtype=np.float64)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract poses (x, y, yaw) from data/data.db3")
    parser.add_argument("db_path", nargs="?", default="data/data.db3", help="Path to bag dir or .db3 file")
    parser.add_argument("-t", "--topic", default=DEFAULT_POSE_TOPIC, help="Pose topic name")
    parser.add_argument("-o", "--output", default="data/poses.npy", help="Output .npy path (default: data/poses.npy)")
    args = parser.parse_args()

    poses = extract_poses(db_path=args.db_path, topic=args.topic, output_path=args.output)
    print(f"Extracted {len(poses)} poses, shape {poses.shape}")
    print(f"Saved to {args.output}")
