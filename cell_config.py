import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
class cell ():
    def __init__(self,Barrier, exit_Vertices,vrt ):
        self.bar = Barrier
        # self.wrd = world
        self.exit_vrt = exit_Vertices
        self.vrt = np.array(vrt)
        self.x_min = np.min(self.vrt[:,0])
        self.x_max = np.max(self.vrt[:,0])
        self.y_min = np.min(self.vrt[:,1])
        self.y_max = np.max(self.vrt[:,1])
    def plot_cell(self, ax):
        for i in range(len(self.vrt)-1):
            ax.plot([self.vrt[i,0], self.vrt[i+1,0]], [self.vrt[i,1], self.vrt[i+1,1]], color = 'black')
        
        ax.plot([self.vrt[0,0], self.vrt[-1,0]], [self.vrt[0,1], self.vrt[-1,1]], color = 'black')
        for wall in (self.bar):
            ax.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], color = 'red')
        if len(self.exit_vrt) > 0:
            ax.plot([self.exit_vrt[0][0], self.exit_vrt[1][0]], [self.exit_vrt[0][1], self.exit_vrt[1][1]], color = 'green')
    def check_in_polygon(self, p):
            """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
            from scipy.spatial import Delaunay
            p = np.reshape(p,(1,2))
            
            if not isinstance(self.vrt,Delaunay):
                hull = Delaunay(self.vrt)

            return (hull.find_simplex(p)>=0)[0]
        


# xmin1 = -0.357
# xmax1 = 3.1
# ymin1 = -2.0
# ymax1 = -0.1

# xmax2 = xmax1
# ymax2 = ymin1

# xmin2 = 0.5
# ymin2 = -5.7
##vertices of the L-shape environment
# p0 = np.array([xmin1, ymax1])
# p1 = np.array([xmin1, ymin1])
# p2 = np.array([xmin2, ymax2])
# p3 = np.array([xmin2, ymin2])
# p4 = np.array([xmax2, ymin2])
# p5 = np.array([xmax1, ymax1])


xmin1 = -0.56
xmax1 = 3.1
ymin1 = -1.6
ymax1 = 1

xmax2 = xmax1
ymax2 = ymin1

xmin2 = 1.6
ymin2 = -3.1
margin = 0.5
##positions based on the LiDAR data
# p0 = np.array([-0.56, 1])
# p1 = np.array([-0.56, -1.6])
# p2 = np.array([1.6, -1.6])
# p3 = np.array([1.6, -3.1])
# p4 = np.array([3.7, -3.1])
# p5 = np.array([3.7, 1])

###Postions based on motion capture and moving the robot manually
p0 = np.array([0, 0.1])
p1 = np.array([0, -1.3])
p2 = np.array([1.8 ,-1.3])
p3 = np.array([1.8, -3.6])
p4 = np.array([3.2, -3.6])
p5 = np.array([3.2, 0.1])

# p0 = np.array([xmin1, ymax1])
# p1 = np.array([xmin1, ymin1])
# p2 = np.array([xmin2, ymax2])
# p3 = np.array([xmin2, ymin2])
# p4 = np.array([xmax2, ymin2])
# p5 = np.array([xmax1, ymax1])

### Middle points for convex regions (for 10 equal square cells)
# m1 = 0.5*(p0+p5)
m1 = np.array([p2[0], p0[1]])
m2 = 0.5*(p0+p1)
m3 = 0.5*(p1+p2)
m4 = 0.5*(p2+p3)
m5 = 0.5*(p3+p4)
m6 = np.array([p5[0], p2[1]])
m7 = 0.5*(m6+p4)
m8 = 0.5*(m6+p5)
m9 = 0.5*(m1+p5)
m10 = 0.5*(p0+m1)
m11 = 0.5*(m10+m3)
m12 = 0.5*(m1+p2)
m13 = 0.5*(m12+m8)
m14 = np.array([0.5*(p2[0]+p4[0]), p2[1]])

m15 = 0.5*(m4+m7)

# m0 = np.array([xmin2, ymax1])
# m1 = 0.5*(m0+p5)
# m2 = 0.5*(p5+p4)
# m3 = 0.5*(p4+p3)
# m4= 0.5*(m3+m1)

def plot_m_points(ax):
    """
    Plot all m points (m1-m15) with their labels on the given axes.
    
    Parameters:
    ax: matplotlib axes object to plot on
    """
    # Collect all m points and their names
    m_points = {
        'm1': m1, 'm2': m2, 'm3': m3, 'm4': m4, 'm5': m5,
        'm6': m6, 'm7': m7, 'm8': m8, 'm9': m9, 'm10': m10,
        'm11': m11, 'm12': m12, 'm13': m13, 'm14': m14, 'm15': m15, 'p0': p0, 'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4, 'p5': p5
    }

    # m_points = {
    #     'm0': m0, 'm1': m1, 'm2': m2, 'm3': m3, 'm4': m4,
    #     'p0': p0, 'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4, 'p5': p5
    # }
    
    # Plot each point and add label
    for name, point in m_points.items():
        ax.plot(point[0], point[1], 'bo', markersize=8)  # Blue circle marker
        ax.text(point[0], point[1], f'  {name}', fontsize=9, 
                verticalalignment='bottom', horizontalalignment='left')


def plot_data_points(directory, ax, color='blue', marker='o', markersize=3, alpha=0.6):
    """
    Plot data points from JSON files in a given directory.
    
    Parameters:
    directory: Path to directory containing JSON files
    ax: matplotlib axes object to plot on
    color: Color for the data points (default: 'blue')
    marker: Marker style for the points (default: 'o')
    markersize: Size of the markers (default: 3)
    alpha: Transparency of the markers (default: 0.6)
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"Warning: Directory '{directory}' does not exist.")
        return
    
    if not dir_path.is_dir():
        print(f"Warning: '{directory}' is not a directory.")
        return
    
    # Get all JSON files in directory
    json_files = list(dir_path.glob("*.json"))
    
    if len(json_files) == 0:
        print(f"No JSON files found in '{directory}'.")
        return
    
    # Collect all positions
    positions = []
    valid_count = 0
    invalid_count = 0
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            position = data.get("position", [])
            
            if len(position) >= 2:
                x, y = position[0], position[1]
                positions.append([x, y])
                valid_count += 1
            else:
                invalid_count += 1
        
        except Exception as e:
            invalid_count += 1
            continue
    
    if len(positions) == 0:
        print(f"No valid position data found in JSON files in '{directory}'.")
        return
    
    # Convert to numpy array for plotting
    positions = np.array(positions)
    
    # Plot all points
    ax.scatter(positions[:, 0], positions[:, 1], c=color, marker=marker, 
               s=markersize, alpha=alpha, label=f'Data points ({valid_count})')
    
    print(f"Plotted {valid_count} data points from {len(json_files)} JSON files in '{directory}'.")
    if invalid_count > 0:
        print(f"  (Skipped {invalid_count} files with invalid data)")


def plot_pose_from_bag(bag_path, topic_name='/vrpn_client_node/jackal/pose', ax=None, 
                        color='red', marker='o', markersize=2, alpha=0.6, label=None):
    """
    Extract and plot pose points from a ROS2 bag file.
    
    Parameters:
    bag_path: Path to the ROS2 bag directory (containing .db3 file) or direct path to .db3 file
    topic_name: Name of the topic to read from (default: '/vrpn_client_node/jackal/pose')
    ax: matplotlib axes object to plot on (if None, creates a new figure)
    color: Color for the data points (default: 'red')
    marker: Marker style for the points (default: 'o')
    markersize: Size of the markers (default: 2)
    alpha: Transparency of the markers (default: 0.6)
    label: Label for the plot (default: None, will auto-generate)
    
    Returns:
    positions: numpy array of shape (N, 2) containing [x, y] positions
    ax: matplotlib axes object used for plotting
    """
    # Store original path for later use
    original_bag_path = bag_path
    bag_path = Path(bag_path)
    
    # Try using rosbag2_py (official ROS2 API) first
    try:
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
        
        # Handle both directory and direct file paths
        if bag_path.is_dir():
            bag_dir = bag_path
        elif bag_path.is_file() and bag_path.suffix == '.db3':
            bag_dir = bag_path.parent
        else:
            print(f"Error: '{bag_path}' is not a valid bag file or directory.")
            return None, ax
        
        if not bag_dir.exists():
            print(f"Error: Bag directory '{bag_dir}' does not exist.")
            return None, ax
        
        # Set up storage options
        storage_options = StorageOptions()
        storage_options.uri = str(bag_dir)
        storage_options.storage_id = 'sqlite3'
        
        # Set up converter options
        converter_options = ConverterOptions()
        converter_options.input_serialization_format = 'cdr'
        converter_options.output_serialization_format = 'cdr'
        
        # Create reader
        reader = SequentialReader()
        reader.open(storage_options, converter_options)
        
        # Set topic filter
        topic_types = reader.get_all_topics_and_types()
        topic_to_read = None
        for topic_metadata in topic_types:
            if topic_metadata.name == topic_name:
                topic_to_read = topic_metadata
                break
        
        if topic_to_read is None:
            print(f"Error: Topic '{topic_name}' not found in bag file.")
            print(f"Available topics: {[t.name for t in topic_types]}")
            return None, ax
        
        # Read messages
        positions = []
        reader.set_read_filter([topic_name])
        
        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            if topic == topic_name:
                # Deserialize message
                from rclpy.serialization import deserialize_message
                from rosidl_runtime_py.utilities import get_message
                
                PoseStamped = get_message('geometry_msgs/msg/PoseStamped')
                msg = deserialize_message(data, PoseStamped)
                
                # Extract x, y from pose
                x = msg.pose.position.x
                y = msg.pose.position.y
                positions.append([x, y])
        
        if len(positions) == 0:
            print(f"Warning: No messages found for topic '{topic_name}'.")
            return None, ax
        
        positions = np.array(positions)
        print(f"Extracted {len(positions)} pose points from topic '{topic_name}'.")
        
    except ImportError:
        # Fallback to SQLite approach if rosbag2_py is not available
        print("rosbag2_py not available, trying SQLite approach...")
        try:
            from rclpy.serialization import deserialize_message
            from rosidl_runtime_py.utilities import get_message
            import sqlite3
            
            # Handle both directory and direct file paths
            if bag_path.is_dir():
                db3_files = list(bag_path.glob("*.db3"))
                if len(db3_files) == 0:
                    print(f"Error: No .db3 file found in directory '{bag_path}'.")
                    return None, ax
                db3_path = db3_files[0]
            elif bag_path.is_file() and bag_path.suffix == '.db3':
                db3_path = bag_path
            else:
                print(f"Error: '{bag_path}' is not a valid bag file or directory.")
                return None, ax
            
            if not db3_path.exists():
                print(f"Error: Bag file '{db3_path}' does not exist.")
                return None, ax
            
            # Connect to SQLite database
            conn = sqlite3.connect(str(db3_path))
            cursor = conn.cursor()
            
            # Get message type
            PoseStamped = get_message('geometry_msgs/msg/PoseStamped')
            
            # Query messages from the topic
            cursor.execute("SELECT id FROM topics WHERE name = ?", (topic_name,))
            topic_row = cursor.fetchone()
            
            if topic_row is None:
                print(f"Error: Topic '{topic_name}' not found in bag file.")
                conn.close()
                return None, ax
            
            topic_id = topic_row[0]
            
            # Get all messages for this topic
            cursor.execute("""
                SELECT timestamp, data 
                FROM messages 
                WHERE topic_id = ? 
                ORDER BY timestamp
            """, (topic_id,))
            
            rows = cursor.fetchall()
            
            if len(rows) == 0:
                print(f"Warning: No messages found for topic '{topic_name}'.")
                conn.close()
                return None, ax
            
            # Extract positions
            positions = []
            for timestamp, data in rows:
                try:
                    msg = deserialize_message(data, PoseStamped)
                    x = msg.pose.position.x
                    y = msg.pose.position.y
                    positions.append([x, y])
                except Exception as e:
                    print(f"Warning: Failed to deserialize message at timestamp {timestamp}: {e}")
                    continue
            
            conn.close()
            
            if len(positions) == 0:
                print("Error: No valid pose messages could be extracted.")
                return None, ax
            
            positions = np.array(positions)
            print(f"Extracted {len(positions)} pose points from topic '{topic_name}'.")
            
        except Exception as e:
            print(f"Error: Failed to read bag file. {e}")
            print("Please ensure ROS2 is properly installed and sourced, or install rosbag2_py.")
            return None, ax
    
    except Exception as e:
        print(f"Error reading bag file: {e}")
        return None, ax
    
    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Pose trajectory from {Path(original_bag_path).name}')
        ax.grid(True)
    
    # Plot points
    if label is None:
        label = f'Pose trajectory ({len(positions)} points)'
    
    ax.scatter(positions[:, 0], positions[:, 1], c=color, marker=marker,
               s=markersize, alpha=alpha, label=label)
    
    print(f"Plotted {len(positions)} pose points from bag file.")
    
    return positions, ax


def plot_vector_field(ax, pos_file='pos_ls.npy', control_file='control_ls.npy', 
                      scale=1.0, color='blue', alpha=0.7, width=0.003):
    """
    Load position and control data from .npy files and plot a vector field.
    
    Parameters:
    ax: matplotlib axes object to plot on
    pos_file: Path to .npy file containing positions (Nx2 array with [x, y] coordinates)
    control_file: Path to .npy file containing control actions (Nx2 array with [ux, uy] vectors)
    scale: Scaling factor for the vectors (default: 1.0)
    color: Color for the vectors (default: 'blue')
    alpha: Transparency of the vectors (default: 0.7)
    width: Width of the arrow shafts (default: 0.003)
    """
    # Load position data
    try:
        pos_ls = np.load(pos_file)
    except FileNotFoundError:
        print(f"Error: Position file '{pos_file}' not found.")
        return
    except Exception as e:
        print(f"Error loading position file '{pos_file}': {e}")
        return
    
    # Load control data
    try:
        control_ls = np.load(control_file)
        control_ls = control_ls.squeeze()
    except FileNotFoundError:
        print(f"Error: Control file '{control_file}' not found.")
        return
    except Exception as e:
        print(f"Error loading control file '{control_file}': {e}")
        return
    
    # Reshape if needed (e.g., from (N, 2, 1) to (N, 2))
    if pos_ls.ndim == 3 and pos_ls.shape[2] == 1:
        pos_ls = pos_ls.squeeze(axis=2)
    elif pos_ls.ndim == 3:
        pos_ls = pos_ls.reshape(-1, 2)
    
    if control_ls.ndim == 3 and control_ls.shape[2] == 1:
        control_ls = control_ls.squeeze(axis=2)
    elif control_ls.ndim == 3:
        control_ls = control_ls.reshape(-1, 2)
    
    # Check shapes
    if pos_ls.ndim != 2 or pos_ls.shape[1] != 2:
        print(f"Error: Position data should be Nx2 array, got shape {pos_ls.shape}")
        return
    
    if control_ls.ndim != 2 or control_ls.shape[1] != 2:
        print(f"Error: Control data should be Nx2 array, got shape {control_ls.shape}")
        return
    
    # Check that lengths match
    if len(pos_ls) != len(control_ls):
        print(f"Warning: Position and control arrays have different lengths ({len(pos_ls)} vs {len(control_ls)}). Using minimum length.")
        min_len = min(len(pos_ls), len(control_ls))
        pos_ls = pos_ls[:min_len]
        control_ls = control_ls[:min_len]
    
    # Extract x, y positions
    x_pos = pos_ls[:, 0]
    y_pos = pos_ls[:, 1]
    
    # Extract ux, uy control vectors
    ux = control_ls[:, 0] * scale
    uy = control_ls[:, 1] * scale
    
    # Plot vector field using quiver
    ax.quiver(x_pos, y_pos, ux, uy, angles='xy', scale_units='xy', 
              color=color, alpha=alpha, width=width)
    
    print(f"Plotted vector field with {len(pos_ls)} vectors from '{pos_file}' and '{control_file}'.")


c_Lshape = cell(
    Barrier=[],
    exit_Vertices=[],
    vrt=[p0, p1, p2, p3, p4, p5])



# c_Lshape0 = cell(
#     Barrier=[],
#     exit_Vertices=[],
#     vrt=[p00, p10, p20, p30, p40, p50])


##old cells
# plt.show()
# c0 = cell( 
#     Barrier=[
#         [p1,p2],
#         [m0, p0],
#         [p1, p0]
#     ],
#     exit_Vertices=[m0, p2],
#     vrt=[p0, p1, p2, m0]
# )

# c1 = cell(
#     Barrier=[
#         [m4,m1],
#         [m1, p5],
#         [p5, m2]
#     ],
#     exit_Vertices=[m4, m2],
#     vrt=[m4, m1, p5, m2]
# )
# c2 = cell(
#     Barrier=[
#         [m2,m4],
#         [m2, p4],
#         [p4, m3]
#     ],
#     exit_Vertices=[m3, m4],
#     vrt=[m3, m4, m2, p4]
# )

##new cells
delta = 0.001
c0 = cell(
    Barrier=[
        [p0,m2],
        [m2, m11+np.array([0, delta])],
        [m10+np.array([0, -delta]), p0]
    ],
    exit_Vertices=[m11, m10],
    vrt=[p0, m2, m11, m10]
)
c1 = cell(
    Barrier=[
        [m1 - np.array([0, delta]),m10],
        [m10, m11],
        [m11, m12+np.array([0, delta])]
    ],
    exit_Vertices=[m1, m12],
    vrt=[m10, m11, m12, m1]
)
c2 = cell(
    Barrier=[
        [m12,m11-np.array([0, delta])],
        [m12, p2],
        [p2, m3+np.array([0, delta])]
    ],
    exit_Vertices=[m3, m11],
    vrt=[m3, m11, m12, p2]
)
c3 = cell(
    Barrier=[
        [m11 + np.array([-delta, 0]),m3],
        [m3, p1],
        [p1, m2+ np.array([delta, 0])]
    ],
    exit_Vertices=[m11, m2],
    vrt=[m11, m3, p1, m2]
)
c4 = cell(
    Barrier=[
        [m13+ np.array([0, delta]),m12],
        [m12, m1],
        [m1, m9-np.array([0, delta])]
    ],
    exit_Vertices=[m9, m13],
    vrt=[m13, m12, m1, m9]
)
c5 = cell(  
    Barrier=[
        [m9,m13+ np.array([delta, 0])],
        [m9, p5],
        [p5, m8 + np.array([-delta, 0])]
    ],
    exit_Vertices=[m8, m13],
    vrt=[m9, m13, m8, p5]
)

c6 = cell(
    Barrier=[
        [m13,m8],
        [m8, m6 + np.array([-delta, 0])],
        [m13, m14+ np.array([delta, 0])]
    ],
    exit_Vertices=[m14, m6],
    vrt=[m6, m8, m13, m14]
)
c7 = cell(
    Barrier=[
        [m7 + np.array([-delta, 0]),m6],
        [m6, m14],
        [m14, m15+ np.array([delta, 0])]
    ],
    exit_Vertices=[m7, m15],
    vrt=[m7, m6, m14, m15]
)
c8 = cell(
    Barrier=[
        [m15+ np.array([0, -delta]),m7],
        [m7, p4],
        [p4, m5+ np.array([0, delta])]
    ],
    exit_Vertices=[m15, m5],
    vrt=[m15, m7, p4, m5]
)
c9 = cell(
    Barrier=[
        [m15 -np.array([delta, 0]), m5],
        [m5, p3],
        [p3, m4+ np.array([delta, 0])]
    ],
    exit_Vertices=[m15, m4],
    vrt=[m15, m5, p3, m4]
)
c10 = cell(
    Barrier=[
        [m14 - np.array([delta, 0]),m15],
        [m15, m4],
        [m4, p2 +np.array([delta, 0])]
    ],
    exit_Vertices=[p2, m14],
    vrt=[m14, m15, m4, p2]
)

c11 = cell(
    Barrier=[
        [p2+np.array([0, delta]),m14],
        [m14, m13],
        [m13, m12+np.array([0, -delta])]
    ],
    exit_Vertices=[m12, p2],
    vrt=[m12, m13, m14, p2]
)

cell_ls = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11]
if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    # c_Lshape0.plot_cell(ax)
    c_Lshape.plot_cell(ax)
    plot_m_points(ax)
    c0.plot_cell(ax)
    c1.plot_cell(ax)
    c2.plot_cell(ax)
    c3.plot_cell(ax)
    c4.plot_cell(ax)
    c5.plot_cell(ax)
    c6.plot_cell(ax)
    c7.plot_cell(ax)
    c8.plot_cell(ax)
    c9.plot_cell(ax)
    c10.plot_cell(ax)
    c11.plot_cell(ax)
    # plot_pose_from_bag('test_2026-01-27-15-49-04', ax=ax, color='blue', markersize=3)
    # plot_vector_field(ax, pos_file='pos_ls_bag.npy', control_file='control_ls_bag.npy')
    plot_vector_field(ax, pos_file='trj_data/pos_ls.npy', control_file='trj_data/control_ls.npy')
    # plot_vector_field(ax, pos_file='rosbags/test_2026-02-19-11-53-31/trj_data_on_robot/pos_ls.npy', control_file='rosbags/test_2026-02-19-11-53-31/trj_data_on_robot/control_ls.npy')
    
    # plot_data_points('/home/mehdi/lidardata/cells_kernels/c0/deg90', ax)
    # plot_data_points('data_deg/deg90', ax)
    # plot_data_points('lidardata_part1', ax)
    # plot_data_points('lidardata_part2', ax)
    plt.show()