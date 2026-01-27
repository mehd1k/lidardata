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
    # def check_in_polygon(self, p):
    #         """
    #     Test if points in `p` are in `hull`

    #     `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    #     `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    #     coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    #     will be computed
    #     """
    #         p = np.reshape(p,(1,2))
            
    #         if not isinstance(self.vrt,Delaunay):
    #             hull = Delaunay(self.vrt)

    #         return (hull.find_simplex(p)>=0)[0]
        


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
p2 = np.array([2 ,-1.3])
p3 = np.array([2, -2.8])
p4 = np.array([3.0, -2.8])
p5 = np.array([3.0, 0.1])

# p0 = np.array([xmin1, ymax1])
# p1 = np.array([xmin1, ymin1])
# p2 = np.array([xmin2, ymax2])
# p3 = np.array([xmin2, ymin2])
# p4 = np.array([xmax2, ymin2])
# p5 = np.array([xmax1, ymax1])

### Middle points for convex regions (for 10 equal square cells)
# m1 = 0.5*(p0+p5)
m1 = np.array([2, 0.1])
m2 = 0.5*(p0+p1)
m3 = 0.5*(p1+p2)
m4 = 0.5*(p2+p3)
m5 = 0.5*(p3+p4)
m6 = 0.5*(p4+p5)
m7 = 0.5*(m6+p4)
m8 = 0.5*(m6+p5)
m9 = 0.5*(m1+p5)
m10 = 0.5*(p0+m1)
m11 = 0.5*(m10+m3)
m12 = 0.5*(m1+p2)
m13 = 0.5*(m12+m8)
m14 = 0.5*(m6+p2)
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
c0 = cell(
    Barrier=[
        [p0,m2],
        [m2, m11],
        [m10, p0]
    ],
    exit_Vertices=[m11, m10],
    vrt=[p0, m2, m11, m10]
)
c1 = cell(
    Barrier=[
        [m1,m10],
        [m10, m11],
        [m1, m12]
    ],
    exit_Vertices=[m11, m12],
    vrt=[m10, m11, m12, m1]
)
c2 = cell(
    Barrier=[
        [m12,m11],
        [m12, p2],
        [p2, m3]
    ],
    exit_Vertices=[m3, m11],
    vrt=[m3, m11, m12, p2]
)
c3 = cell(
    Barrier=[
        [m11,m3],
        [m3, p1],
        [p1, m2]
    ],
    exit_Vertices=[m11, m2],
    vrt=[m11, m3, p1, m2]
)
cell_ls = [c0, c1, c2, c3]
if __name__ == "__main__":
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    # c_Lshape0.plot_cell(ax)
    c_Lshape.plot_cell(ax)
    plot_m_points(ax)
    c0.plot_cell(ax)
    # print('xmin', c0.x_min, 'xmax', c0.x_max, 'ymin', c0.y_min, 'ymax', c0.y_max)
    # c1.plot_cell(ax)
    c2.plot_cell(ax)
    c3.plot_cell(ax)
    plot_data_points('/home/mehdi/lidardata/cells_kernels/c0/deg90', ax)
    # plot_data_points('data_deg/deg90', ax)
    plt.show()