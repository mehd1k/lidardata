"""
Extract images from a ROS 2 bag (SQLite .db3) containing CompressedImage messages.
"""

from pathlib import Path


# Default image topic in data/data.db3
DEFAULT_IMAGE_TOPIC = "/camera/color/image_raw/compressed"


def extract_images(
    db_path="data/data.db3",
    topic=None,
    output_dir=None,
    prefix="frame",
    fmt_extension=None,
):
    """
    Extract images from a ROS 2 bag database (data.db3).

    Supports sensor_msgs/msg/CompressedImage. Uses rosbag2_py when available,
    otherwise falls back to reading the SQLite DB and deserializing with
    rclpy/rosidl_runtime_py (requires a sourced ROS 2 environment).

    Parameters
    ----------
    db_path : str or Path
        Path to the bag directory or to the .db3 file (e.g. "data/data.db3").
    topic : str, optional
        Image topic name. Defaults to "/camera/color/image_raw/compressed".
    output_dir : str or Path, optional
        Directory to save images. Defaults to "data/extracted_images".
    prefix : str
        Filename prefix for saved images (default "frame" -> frame_000000.jpg).
    fmt_extension : str, optional
        File extension override (e.g. "jpg", "png"). If None, uses the
        format from the message (e.g. "jpeg" -> "jpg").

    Returns
    -------
    list of Path
        Paths to the saved image files.

    Raises
    ------
    FileNotFoundError
        If db_path does not exist.
    ValueError
        If the topic is not found or has no CompressedImage messages.
    """
    topic = topic or DEFAULT_IMAGE_TOPIC
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

    if output_dir is None:
        output_dir = db_path.parent / "extracted_images" if db_path.is_file() else db_path / "extracted_images"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _extension_from_format(fmt):
        if fmt_extension:
            return fmt_extension if fmt_extension.startswith(".") else f".{fmt_extension}"
        if not fmt:
            return ".jpg"
        fmt = fmt.strip().lower()
        if fmt in ("jpeg", "jpg"):
            return ".jpg"
        if fmt == "png":
            return ".png"
        return f".{fmt}"

    # Try rosbag2_py first
    try:
        return _extract_with_rosbag2_py(
            bag_dir=bag_dir,
            topic=topic,
            output_dir=output_dir,
            prefix=prefix,
            ext_fn=_extension_from_format,
        )
    except ImportError:
        pass

    # Fallback: SQLite + rclpy deserialization
    try:
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        import sqlite3
    except ImportError as e:
        raise ImportError(
            "Need either rosbag2_py or (rclpy + rosidl_runtime_py) to read the bag. "
            "Source your ROS 2 workspace and try again."
        ) from e

    CompressedImage = get_message("sensor_msgs/msg/CompressedImage")
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

    saved = []
    for idx, (msg_id, timestamp, data) in enumerate(rows):
        try:
            msg = deserialize_message(data, CompressedImage)
        except Exception as e:
            print(f"Warning: skip message id={msg_id} timestamp={timestamp}: {e}")
            continue
        ext = _extension_from_format(getattr(msg, "format", None) or "jpeg")
        path = output_dir / f"{prefix}_{idx:06d}{ext}"
        path.write_bytes(bytes(msg.data))
        saved.append(path)
    print(f"Extracted {len(saved)} images to {output_dir}")
    return saved


def _extract_with_rosbag2_py(bag_dir, topic, output_dir, prefix, ext_fn):
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
    CompressedImage = get_message("sensor_msgs/msg/CompressedImage")
    saved = []
    idx = 0
    while reader.has_next():
        read_topic, data, _ = reader.read_next()
        if read_topic != topic:
            continue
        msg = deserialize_message(data, CompressedImage)
        ext = ext_fn(getattr(msg, "format", None) or "jpeg")
        path = output_dir / f"{prefix}_{idx:06d}{ext}"
        path.write_bytes(bytes(msg.data))
        saved.append(path)
        idx += 1
    print(f"Extracted {len(saved)} images to {output_dir}")
    return saved


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract images from data/data.db3")
    parser.add_argument("db_path", nargs="?", default="data/data.db3", help="Path to bag dir or .db3 file")
    parser.add_argument("-t", "--topic", default=DEFAULT_IMAGE_TOPIC, help="Image topic name")
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory for images")
    parser.add_argument("-p", "--prefix", default="frame", help="Filename prefix")
    args = parser.parse_args()
    extract_images(
        db_path=args.db_path,
        topic=args.topic,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )
