

import numpy as np
import matplotlib.pyplot as plt
import math
# from shapely.geometry import Point, Polygon
from scipy.spatial import Delaunay
# import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
import sys
import scipy.io as sio
import os
from visualize_lidar_scan import generate_occupancy_grid, load_lidar_data, generate_occupancy_grid_polar, visualize_occupancy_grid_polar
# from vaemodel import load_vae_model
from scipy.io import loadmat
from cell_config import cell, cell_ls


def map_address(ix,iy,nx):
    """map address for a matrix of size nx*ny to a  vectorized version index"""
    return iy*nx+ix

def vectorize_matrix(P):
    ny = np.shape(P)[0]
    nx = np.shape(P)[1]
    out = np.zeros((int(nx*ny), 1))
    for iy in range(ny):
        for ix in range(nx):
            out[int(iy*nx+ix)] = P[iy, ix]
    return out





       

class Discrertized_Linear_Controller():
    def __init__(self, A, B, dt):
        self.A = A
        self.B = B
        self.dt = dt
        self.A_dis = np.zeros_like(A)
        self.B_dis = np.zeros_like(B)
    def A_discretized_calculator(self):
        A_dis = np.eye(len(self.A))
        A_n = A_dis
        for n in range(20):
            A_n = self.A@A_n
            new_trem = 1/math.factorial(n)*A_n*self.dt**n
            A_dis = A_dis + new_trem
        self.A_dis = A_dis
    
    
    def B_discretized_calculator(self):
        B_dis = self.B*self.dt
        B_n = B_dis
        for n in range(2,20):
            B_n = self.A@B_n
            new_trem = 1/math.factorial(n)*B_n*self.dt**n
            B_dis = B_dis + new_trem
        self.B_dis = B_dis

    def __call__(self) :
        self.A_discretized_calculator()
        self.B_discretized_calculator()
        return self.A_dis, self.B_dis




def cbf(x_pos, AH, bH):
     c =(AH@x_pos+np.reshape(bH, (-1,1) ))
     if all(c>=0):
         return 1
     else:
         return 0

def polygon(x_pos, Ax, bx):
    c = Ax@x_pos+np.reshape(bx, (-1,1) )
    if all(c<=0):
        return 1
    else:
        return 0




def load_RSC_data():
    dir = 'allocentric_ratemaps/RSC/data'
    files = os.listdir(dir)
    output = []
    for file in files:
        data = np.load(os.path.join(dir, file))
        output.append(data)
    return np.array(output)




class Control_cal():
    def __init__(self, cell, A, B, dt,ch,cv, eps, sigma_max , grid_size_x, grid_size_y , directory_mat, directory_save ,con_lim = 4, measurement_mode = 'neural_lidar'):
        
        self.nx = grid_size_x
        self.ny = grid_size_y
        self.gs = [self.nx,self.ny]
        self.measurement_mode = measurement_mode
        print('***********measurement_mode='+self.measurement_mode+'***********')
        self.cell = cell
        self.RSC_data = load_RSC_data()
       
        self.ch = ch
        self.cv = cv
        self.o = np.reshape(cell.exit_vrt[1],(2,1))
        # self.rate_maps_cell = rate_maps_cell
        
        
        self.vrt = np.array(cell.vrt)
        self.xmin = np.min(self.vrt[:,0])
        self.ymin = np.min(self.vrt[:,1])
        self.xmax = np.max(self.vrt[:,0])
        self.ymax = np.max(self.vrt[:,1])
        
        dx = (self.xmax-self.xmin)
        dy = (self.ymax-self.ymin)
        dmax = max(dx, dy)
        
        self.d = [dx,dy]
        self.Po_ls =[]
        self.center = np.mean(self.vrt, axis=0)
        self.eps = eps
        self.sigma_max = sigma_max
        # self.eps = max(2*dx/nx,2*dy/ny)*1.1
        # self.eps = 10
        # self.sigma_max = self.eps**2
        # self.sigma_max = 160
        self.A = A
        self.B = B
        self.dt = dt
    
        self.l = np.array([[self.xmin], [self.ymin]])
        self.Ax, self.bx = self.get_Axbx()
        
        
        
        self.AH,  self.bH = self.get_Ahbh()

      

        self.v = self.get_v()
        self.grid = []
        self.con_lim = con_lim
        self.directory_save =directory_save
  
        self.kernel, self.x_ls, self.y_ls = self.load_mat_files2(directory_mat)
        self._get_theta()
        self.kernel_ls = [self.kernel]
      
        # self.kernel_ls =  [np.log(np.abs(self.kernel)+1)/np.max(np.abs(self.kernel)), np.sin(self.kernel)]

    
    def load_mat_files2(self, directory):
            if directory == None:
                return
        

       
            data_ls = []
            x_ls = []
            y_ls = []
            # Iterate through the directory and process each .mat file
            for filename in os.listdir(directory):
                if filename.endswith('.json'):
                  
                    
                
                    json_file = os.path.join(directory, filename)
                    data = load_lidar_data(json_file)
                    x = data['position'][0]
                    y = data['position'][1]
                    occupancy_grid_polar, polar_params = generate_occupancy_grid_polar(json_file)
                    measurement = []

                    for i in range(len(self.RSC_data)):
                        measurement.append(np.sum(self.RSC_data[i]*occupancy_grid_polar.T))
                    measurement = np.array(measurement)
                        
                        
                        
               
                    # 
                    # Store the data in a dictionary with keys as (x, y) tuples
                    
                    x_ls.append(x)
                    y_ls.append(y)
                    data_ls.append(measurement)
            # Extract grid parameters from the coordinate values
            # x_coords = sorted(set([coord[0] for coord in grid_data.keys()]))
            # y_coords = sorted(set([coord[1] for coord in grid_data.keys()]))
            
            n_data = len(data_ls)
            print('n_data = ', n_data)  
            # Initialize output kernel matrix (100 neurons x nx*ny grid positions)
            # output = np.zeros([len(measurement), n_data])
            
            # Build kernel by placing each file in the column it maps to
            # Process files in sorted order (by y, then x) so that later files overwrite earlier ones
            # This ensures deterministic behavior when multiple files map to the same column
           

            return np.array(data_ls).T, np.array(x_ls), np.array(y_ls)

    
    def _get_theta(self):
       
        

      
      
        
        self.U = np.array([self.x_ls,self.y_ls])
        
        szp = int(len(self.x_ls))
        ### Making Ap
        self.Ap = np.zeros((4,szp))
        
        
        self.Ap[0:2] = self.U
        self.Ap[2:] = -self.U
        ### Making bp
        self.bp = np.array([ [-self.eps], [-self.eps], [-self.eps], [-self.eps]] )
        self.Ax2 = np.array([ [-1,0],[0,-1], [1,0], [0,1]])
       

    def check_Probability_constraints(self,x_pos, y_pos):
        pos = np.array([[x_pos],[y_pos]])
        P = self.obs.obs(pos)     
        # P_vec = P.reshape([-1,1])
        P_vec = vectorize_matrix(P)
        x_obs = self.U@P_vec
        print('U@P=',x_obs)
        print(self.Ap@P_vec+self.Ax2@pos+self.bp)


        

    
    def plot_cell(self):
        fig, ax = plt.subplots()
        for i in range(len(self.vrt)-1):
            ax.plot([self.vrt[i,0], self.vrt[i+1,0]], [self.vrt[i,1], self.vrt[i+1,1]], color = 'black')
        
        ax.plot([self.vrt[0,0], self.vrt[-1,0]], [self.vrt[0,1], self.vrt[-1,1]], color = 'black')
        for wall in (self.cell.bar):
            ax.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], color = 'red')
        ax.plot([self.cell.exit_vrt[0][0], self.cell.exit_vrt[1][0]], [self.cell.exit_vrt[0][1], self.cell.exit_vrt[1][1]], color = 'green')
    def get_Axbx(self):
        Ax = []
        bx = [] 
      
        for iv in range(len(self.vrt)-1):
            Ai = np.array([-self.vrt[iv+1][1]+self.vrt[iv][1],  self.vrt[iv+1][0]-self.vrt[iv][0]])
            bi = self.vrt[iv][0]*self.vrt[iv+1][1] - self.vrt[iv][1]*self.vrt[iv+1][0]
            
            # check1 = Ai@np.array(self.vrt[iv+1])+bi
            # check2 = Ai@np.array(self.vrt[iv])+bi
            
            if Ai@self.center+bi > 0 :
                Ai = -Ai
                bi = -bi
            Ax.append(Ai)
            bx.append(bi)
        
        
        ###last and first vertices
        Ai = np.array([-self.vrt[-1][1]+self.vrt[0][1],  self.vrt[-1][0]-self.vrt[0][0]])
        bi = self.vrt[0][0]*self.vrt[-1][1] - self.vrt[0][1]*self.vrt[-1][0]
        
        # check1 = Ai@np.array(self.vrt[iv+1])+bi
        # check2 = Ai@np.array(self.vrt[iv])+bi
        
        if Ai@self.center+bi > 0 :
            Ai = -Ai
            bi = -bi
        Ax.append(Ai)
        bx.append(bi)
       
        return np.array(Ax),np.reshape(np.array(bx),(-1,1))
    

    def get_Ahbh(self):
        Ah = []
        bH = [] 
        for seg in self.cell.bar: 
            x1,y1,x2,y2 = seg[0][0],seg[0][1],seg[1][0],seg[1][1]
            Ai = np.array([y1-y2, -(x1-x2)])
            ln = np.linalg.norm(Ai)
            Ai = Ai/ln
            bi = y1*(x1-x2)-x1*(y1-y2)
            bi = bi/ln
            # check1 = Ai@np.array(self.vrt[iv+1])+bi
            # check2 = Ai@np.array(self.vrt[iv])+bi
            
            if Ai@self.center+bi< 0 :
                Ai = -Ai
                bi = -bi
            Ah.append(Ai)
            bH.append(bi)

                
       
  
       
        return np.array(Ah),np.reshape(np.array(bH),(-1,1))
    
    def u(self, measurement):
        # u = self.K@self.kernel@P_vec+self.Kb
        u = self.K@measurement
        u = u.reshape([2,1])
        u += self.Kb
        return u
    


    def u_postion(self, x_pos, y_pos):
        
        
        pos = np.array([[x_pos],[y_pos]])
        P = self.obs.obs(pos)     
        P_vec = P.reshape([-1,1])
        neural_rate = self.kernel@P_vec
        u = self.K@neural_rate+self.Kb
        # con_lim = self.con_lim
        # if np.abs(u[0])>con_lim:
        #     u[0] = con_lim*u[0]/np.abs(u[0])
        # if np.abs(u[1])>con_lim:
        #     u[1] = con_lim*u[1]/np.abs(u[1])
        print('neural rate', neural_rate)
        print('control', u)
        return u
    
    def encode_lidar_with_vae(self, lidar_ranges):
        """
        Encode lidar scan ranges using the VAE model.
        
        Args:
            lidar_ranges: numpy array of lidar ranges (shape: [640] or [N, 640])
        
        Returns:
            numpy array: Encoded latent representation (shape: [latent_dim] or [N, latent_dim])
        """
        if self.VAE is None:
            raise ValueError("VAE model not loaded. Set measurmenet_mode='vae' and ensure model file exists.")
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for VAE encoding")
        
        # Convert to numpy array if needed
        if not isinstance(lidar_ranges, np.ndarray):
            lidar_ranges = np.array(lidar_ranges)
        
        # Handle single sample vs batch
        single_sample = len(lidar_ranges.shape) == 1
        if single_sample:
            lidar_ranges = lidar_ranges.reshape(1, -1)
        
        # Convert to numpy array and ensure float32
        if isinstance(lidar_ranges, torch.Tensor):
            lidar_ranges = lidar_ranges.cpu().numpy()
        lidar_ranges = np.array(lidar_ranges, dtype=np.float32)
        
        # Normalize data (same normalization as training)
        # Normalization parameters should be Python scalars (converted in load_vae_model)
        data_min = float(self.vae_data_min)
        data_max = float(self.vae_data_max)
        
        lidar_normalized = (lidar_ranges - data_min) / (data_max - data_min + 1e-8)
        
        # Convert to torch tensor with float32 dtype and move to device
        lidar_tensor = torch.from_numpy(lidar_normalized).float().to(self.vae_device)
        
        # Encode to latent space
        with torch.no_grad():
            mu, logvar = self.VAE.encode(lidar_tensor)
            # Use mean of latent distribution (or use reparameterize for sampling)
            z = mu  # Or use: z = self.VAE.reparameterize(mu, logvar) for sampling
        
        # Convert back to numpy
        z_numpy = z.cpu().numpy()
        
        # Return single sample or batch
        if single_sample:
            return z_numpy[0]
        else:
            return z_numpy
    


    def check_P_U(self, x_pos, y_pos):
        
        
        pos = np.array([[x_pos],[y_pos]])
        P = self.obs.obs(pos)     
        P_vec = P.reshape([-1,1])
        x_obs = self.U@P_vec
        print(x_obs)
       
    
    
   
    def vector_F(self):

       
        # obs = obs = observation(self.l, self.eps, self.sigma_max, [10,10], self.d)
        CD = Discrertized_Linear_Controller(self.A, self.B, self.dt)
        A_dis, B_dis = CD()
       
        ux_ls = []
        uy_ls = []
        # self.cpox = np.zeros((len(Y),len(X)))
        # self.cpoy = np.zeros((len(Y),len(X)))
        for i_data in range(len(self.y_ls)):
            
    
                
                # Po = obs.obs(x_old).T.flatten().reshape([-1,1])
                u =self.u(self.kernel_ls[0][:,i_data])
                uc = np.copy(u)/np.linalg.norm(u)
            
                
                ux_ls.append(uc[0]) 
                uy_ls.append(uc[1])
             
               
            
        fig, ax = plt.subplots()        
        for i in range(len(self.vrt)-1):
            ax.plot([self.vrt[i,0], self.vrt[i+1,0]], [self.vrt[i,1], self.vrt[i+1,1]], color = 'black')
        
        ax.plot([self.vrt[0,0], self.vrt[-1,0]], [self.vrt[0,1], self.vrt[-1,1]], color = 'black')
        for wall in (self.cell.bar):
            ax.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]], color = 'red')
        ax.plot([self.cell.exit_vrt[0][0], self.cell.exit_vrt[1][0]], [self.cell.exit_vrt[0][1], self.cell.exit_vrt[1][1]], color = 'green')
        sc =1
        ux_ls = sc*np.array(ux_ls)
        uy_ls = sc* np.array(uy_ls)
        ax.quiver(self.x_ls,self.y_ls,ux_ls,uy_ls,angles='xy', scale_units='xy')
        # print('u_min', np.min(np.abs(ux_ls)), np.min(np.abs(uy_ls)))
        # np.save('ux_ls.npy',ux_ls)
        # np.save('uy_ls.npy',uy_ls)
        fig.show()
        # fname = 'vec_field_mod'+str(i_cell)+'ch_'+str(self.ch)+'cv_'+str(self.cv)+'es_'+str(self.eps)+'_'+str(self.sigma_max)+'.png'
        fname = os.path.join(self.directory_save, 'vec_field.png')
        plt.show()
        fig.savefig(fname, dpi = 500)
    
    def check_AxAH(self):
        ny = self.nx
        nx = self.ny
        Cx = np.zeros((ny,nx))
        Ccbf = np.zeros((ny,nx))
        
        X = np.linspace(np.min(self.vrt[:,0])*0.9,np.max(self.vrt[:,0])*1.1,nx)
        Y = np.linspace(np.min(self.vrt[:,1])*0.9,np.max(self.vrt[:,1])*1.1, ny)
        for iy in range(len(Y)):
            for ix in range((len(X))):
                x_pos = np.array([ [X[ix]], [Y[iy]]] )
                # if all(Ax@x_pos+bx<=0):
                Cx[iy,ix] = polygon(x_pos,self.Ax,self.bx)
                Ccbf[iy,ix] = np.reshape(cbf(x_pos, self.AH, self.bH),[1,-1])
                # else:
                #     Cx[iy,ix] = polygon(x_pos,Ax,bx)
                #     Ccbf[:,iy,ix] = np.nan*np.ones_like(Ccbf[:,iy,ix])
                
        Ccbf = np.flip(Ccbf, 0)
        Cx = np.flip(Cx, 0)
        print('*****************CBF**************************')
        print(Ccbf)
        
                
        print('*****************Cx**************************')
        print(Cx)
        
        
    def get_v(self):
        ###normal vector to the exit face
        t_exit = self.cell.exit_vrt[0]-self.cell.exit_vrt[1]
        n_exit = np.array([-t_exit[1], t_exit[0]])
        V = n_exit @(self.center- self.cell.exit_vrt[0])
        if V < 0 :
            n_exit = -n_exit
        return np.reshape(np.array(n_exit)/np.linalg.norm(n_exit), (1,2))
    
    
    def check_in_polygon(self, p):
        """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
        p = np.reshape(p,(1,2))
        from scipy.spatial import Delaunay
        if not isinstance(self.vrt,Delaunay):
            hull = Delaunay(self.vrt)

        return (hull.find_simplex(p)>=0)[0]
    def gauss_kernel(self, mean,var):
        nx,ny = self.nx, self.ny
        dx= self.d[0]
        dy= self.d[1]
    
        Po = np.zeros((ny,nx))
        
        for iy in range(ny):
            for ix in range(nx):
                x_g = ix*2*self.d[0]/(nx-1)-self.d[0]
                y_g = self.d[1] -iy*2*self.d[1]/(ny-1)
                Po[iy,ix] = np.exp(-1/(2*var)*( (x_g-mean[0])**2+(y_g-mean[1])**2))

        Po = Po/np.sum(Po)
        self.Po_ls.append(Po)
        
        plt.imshow(Po)
        plt.show()
        
        tx = np.zeros((ny,nx))
        ty = np.zeros((ny,nx))
        for ix in range(nx):
            
                ty[:, ix] = Po[:,ix]

      
        for iy in range(ny):
            tx[iy, :] = Po[iy,:]
               

        txT = vectorize_matrix(tx)
        txT = np.reshape(txT, [1,-1])

        tyT = vectorize_matrix(ty)
        tyT = np.reshape(tyT, [1,-1])
       
        out = np.array([txT[0],tyT[0]])
        return out
    
    def visualize_kernels(self):
        colors = [
        np.array([1, 0, 0]),  # Red
        np.array([0, 1, 0]),  # Green
        np.array([0, 0, 1]),  # Blue
        np.array([1, 1, 0]),  # Yellow
        np.array([1, 0, 1]),  # Magenta
        np.array([0, 1, 1])   # Cyan
            ]
        # np.random.seed(42)  # Seed for reproducibility, you can remove this if you want truly random colors each time
        # colors = np.random.rand(len(self.Po_ls), 3)  # Generating a unique color (RGB) for each matrix

        # Normalize matrices to ensure values are within [0, 1] for valid color coding
        max_value = np.max([np.max(matrix) for matrix in self.Po_ls])

        # Initialize an empty matrix for the sum
        summed_image = np.zeros((*self.Po_ls[0].shape, 3))  # Assuming all matrices have the same shape

        # Loop through each matrix and its corresponding color
        for matrix in self.Po_ls:
            color_id = np.random.randint(0, len(colors))
            color = colors[color_id]
            # Normalize and color code the current matrix
            colored_matrix = (matrix / max_value)[:, :, None] * color[None, None, :]
            # Add the color-coded matrix to the sum
            summed_image += colored_matrix

        # Display the resulting image
        summed_image = np.clip(summed_image, 0, 1)
        plt.imshow(summed_image)
        plt.axis('off')  # Hide axis for better visualization
        plt.show()
        plt.savefig('kv'+str(len(self.Po_ls))+'.png')
        

    def load_kernels(self, directory):
        # Get the list of all files and directories
        self.kernel_ls = []
        all_files_and_dirs = os.listdir(directory)

        # Filter the list to get only files
        files = [f for f in all_files_and_dirs if os.path.isfile(os.path.join(directory, f))]
        files.sort()
        # print(files)
        for f in files:
            path = os.path.join(directory,f)
            data = scipy.io.loadmat(path)
            self.kernel_ls.append(data['output_resize'].flatten())

        self.kernel_ls = np.array(self.kernel_ls) 
        

      
    def get_K(self,cbf_lb= 0, clf_lb = 0): 
        print('wh', )
        wh = 10**2
        wv =10**-1
        print('wh', wh)
        # nx,ny = self.gs[0],self.gs[1]
        
        szp = int(len(self.x_ls))
        
        szv = len(self.vrt)
       
        Mxv = self.v@self.A+self.cv*self.v
        Mxv = np.reshape(Mxv, [1,-1])
        
        # lim =3*10


        # kernel_ls = self.kernel_ls
        
        ncbf =  len(self.bH)
        m = gp.Model()
        # ###Defining the Optimization problem

        control_lim = 10**-5
       
        


        

        # kernel_ls =[self.U]
        n_kernels = len(self.kernel_ls)
        # K = m.addMVar((2,nk), ub = control_lim, lb = -control_lim, name = 'K')
        nk = len(self.kernel_ls[0])
        K = m.addMVar((n_kernels,2,nk), ub = control_lim, lb = -control_lim, name = 'K')

        control_lim_b = 5
        Kb = m.addMVar((2,1), ub = control_lim_b, lb = -control_lim_b, name = 'Kb')
        infval = 10**6
       
        # Kp = m.addMVar((2,szp), ub = control_lim, lb = -control_lim, name = 'Kp')
        dv = m.addVar(ub = clf_lb, lb = -infval, name = 'dv')
        dh = m.addVar(ub = cbf_lb, lb = -infval, name = 'dh')
        
        # m.addConstr(dh<= dv)
   



        

        #### CLF related vars
       
        ls = m.addVar(name = 'lambda_s_v')
        lp = m.addMVar((1,len(self.bp)), ub = infval, lb = 0, name = 'lambda_p_v')

       
        lx = m.addMVar((1,len(self.bx)), lb= 0, ub = infval, name='lambda_x_v')
        
       

        # ###########

        
        
        ex = np.array([[1,0]])
        ey = np.array([[0,1]])
        tempv = (self.v@self.B)
        Mpv = 0
        
        # Mpv = np.kron(tempv, self.kernel.T) @ K.reshape(-1)
        for i in range(n_kernels):
            Mpv += np.kron(tempv, self.kernel_ls[i].T) @ K[i].reshape(-1)
        xj = np.reshape(self.cell.exit_vrt[0],(2,1))
        rv= -(self.cv*self.v@xj)+self.v@Kb
        
        ########CLF Constraints
        
       
        m.addConstr(-Mxv+lp@self.Ax2+lx@self.Ax== 0)
        m.addConstr((-lx@self.bx)[0,0] -(lp@self.bp)[0,0]+ls+rv[0][0]<= dv)
              
        for ix in range(szp):
            m.addConstr(0<= -Mpv[ix]+(lp@self.Ap)[0,ix]+ls)
           
        
         ###########CBF vars
      
        lsh = m.addMVar(ncbf, name = 'lambda_s_h')
        lph = m.addMVar((ncbf,len(self.bp)), ub = infval, lb = 0, name = 'lambda_p_h')

        
        lxh = m.addMVar((ncbf,len(self.bx)), lb= 0, ub = infval, name='lambda_x_h')
        ## rhoh = m.addMVar((ncbf,4,szp), lb= 0, ub = infval, name='rho_h')
      
        
        ## btxh = m.addMVar((ncbf,szp,len(self.bx)), lb= 0, ub = infval, name='beta_x_h')
        ## etah = m.addMVar((ncbf,szp,2,2), lb = 0, ub =infval, name ='eta_h')

        ########CBF constraints

        for ih in range(ncbf):
        # for ih in [2,1]:
        ## ih = 0
            rh = -self.ch*self.bH[ih] -self.AH[ih]@self.B@Kb
            
            Mxh = -self.AH[ih]@self.A-self.ch*self.AH[ih]
            temph = (-np.array([self.AH[ih]])@self.B)
            
            
            Mph = 0
            # Mph = np.kron(temph, self.kernel.T) @ K.reshape(-1) 
            for i in range(n_kernels):
                Mph += np.kron(temph, self.kernel_ls[i].T) @ K[i].reshape(-1)

            
            m.addConstr(-Mxh+lph[ih]@self.Ax2+lxh[ih]@self.Ax == 0)
            
            
            ##m.addConstr(-Mxh+lph[ih]@self.Ax2+lxh[ih]@self.Ax+((rhoh[ih,0]-rhoh[ih,1])@np.ones(szp)*ex)[0]+((rhoh[ih,2]-rhoh[ih,3])@np.ones(szp)*ey)[0] == 0)
            ##m.addConstr(rhoh[ih,0]+rhoh[ih,1]==0)
            ##m.addConstr(rhoh[ih,2]+rhoh[ih,3]==0)
            

            m.addConstr(-(lxh[ih]@self.bx)[0]-lph[ih]@self.bp+lsh[ih]+rh<= dh)
            for ix in range(szp):
                m.addConstr(0<= -Mph[ix]+(lph[ih]@self.Ap)[ix]+lsh[ih])
                ##m.addConstr(btxh[ih][ix]@self.Ax+((etah[ih][ix][0,0]-etah[ih][ix][0,1])*ex)[0]+((etah[ih][ix][1,0]-etah[ih][ix][1,1])*ey)[0] ==0)
                ##m.addConstr(lzh[ih][0]-etah[ih][ix][0,0]-etah[ih][ix][0,1]==0)
                ##m.addConstr(lzh[ih][1]-etah[ih][ix][1,0]-etah[ih][ix][1,1]==0)
        
   
        m.update()
        m.setObjective(wv*dv+wh*dh,  GRB.MINIMIZE)
        m.update()
        # m.params.NonConvex = 2
        m.optimize()
        if m.Status == GRB.OPTIMAL:
            print('K=', K.X)
            print('K shape', K.X.shape)
            self.K = K.X

            # self.Mp =kernel_sum.X
        
            # self.Mp =0
            # for i in range(len(kernel_ls)):
            #     self.Mp += K[i].X@kernel_ls[i]
            # # print('control gain', K.X)
            print('Kb', Kb.X)
            print('dv,dh =' ,dv.X,dh.X)
            self.Kb = Kb.X
           
            np.save(os.path.join(self.directory_save, 'K.npy'), self.K)
            np.save(os.path.join(self.directory_save, 'Kb.npy'), self.Kb)


    def get_K_old(self):
        ###calcualting gains with LP


        wh =1
        wv =1
        nx,ny = self.gs[0],self.gs[1]
        xj = self.l
        szp = int(nx*ny)

        szv = len(self.vrt)

        Mxv = self.v@self.A+self.cv*self.v
        Mxv = np.reshape(Mxv, [1,-1])

        # lim =3*10

        ncbf =  len(self.bH)
        m = gp.Model()
        # ###Defining the Optimization problem

        control_lim = 5
        
        nk = 100

        
        K = m.addMVar((2,nk), ub = control_lim, lb = -control_lim, name = 'K')
        # K2 = m.addMVar((2,2), ub = control_lim, lb = -control_lim, name = 'K2')
        # Kcos = m.addMVar((2,2), ub = control_lim, lb = -control_lim, name = 'Kcos')
        # Ksin = m.addMVar((2,2), ub = control_lim, lb = -control_lim, name = 'Ksin')
        infval = 10**6
        Kb = m.addMVar((2,1), ub = control_lim, lb = -control_lim, name = 'Kb')
        # Kp = m.addMVar((2,szp), ub = control_lim, lb = -control_lim, name = 'Kp')
        dv = m.addVar(ub = -0.01, lb = -infval, name = 'dv')
        dh = m.addVar(ub = -0.01, lb = -infval, name = 'dh')
        # m.addConstr(dh<= dv)
        # kernel_sum = K@self.U
        # kernel_sum = K@self.U+K2@self.U**2+Kcos@np.cos(self.U)
        
        

        lz = m.addMVar(2, ub = infval, lb =0)
        ls = m.addVar(name = 'lambda_s_v')
        lp = m.addMVar((1,len(self.bp)), ub = infval, lb = 0, name = 'lambda_p_v')


        lx = m.addMVar((1,len(self.bx)), lb= 0, ub = infval, name='lambda_x_v')
        rho = m.addMVar((4,szp), lb= 0, ub = infval, name='rho_v')
        # ry = m.addMVar((2,szp), lb= 0, ub = infval, name='rho_y')

        btx = m.addMVar((szp,len(self.bx)), lb= 0, ub = infval, name='beta_x_v')
        eta = m.addMVar((szp,2,2), lb = 0, ub =infval, name ='eta_v')





        # ###########



        ex = np.array([[1,0]])
        ey = np.array([[0,1]])
        tempv = (self.v@self.B)
        Mpv = np.kron(tempv, self.kernel.T) @ K.reshape(-1)
        # Mpv = tempv@(kernel_sum)
        
            
        xj = np.reshape(self.cell.exit_vrt[0],(2,1))
        rv= -(self.cv*self.v@xj)+self.v@Kb

        ########CLF Constraints

        m.addConstr(-Mxv+lp@self.Ax2+lx@self.Ax+(rho[0]-rho[1])@np.ones(szp)*ex+(rho[2]-rho[3])@np.ones(szp)*ey== 0)
        m.addConstr(rho[0]+rho[1]==0)
        m.addConstr(rho[2]+rho[3]==0)


        m.addConstr((-lx@self.bx)[0,0]+(rho[0]-rho[1])@(self.txT-self.l[0,0]*np.ones((1,szp)))[0]+(rho[2]-rho[3])@(self.tyT-self.l[1,0]*np.ones((1,szp)))[0] -(lp@self.bp)[0,0]+ls+self.sigma_max*lz[0]+self.sigma_max*lz[1]+rv[0][0]<= dv)


        for ix in range(szp):
            m.addConstr(-(btx[ix]@self.bx)[0]+(eta[ix,0,0]-eta[ix,0,1])*(-self.txT[0,ix]+self.l[0,0])+(eta[ix,1,0]-eta[ix,1,1])*(-self.tyT[0,ix]+self.l[1,0])<= -Mpv[ix]+(lp@self.Ap)[0,ix]+ls)
            m.addConstr(btx[ix]@self.Ax+((eta[ix][0,0]-eta[0,1])*ex)[0]+((eta[ix][1,0]-eta[ix][1,1])*ey)[0] ==0)
            m.addConstr(lz[0]-eta[ix][0,0]-eta[ix][0,1]==0)
            m.addConstr(lz[1]-eta[ix][1,0]-eta[ix][1,1]==0)

            ###########CBF vars
        lzh = m.addMVar((ncbf,2), ub = infval, lb =0)
        lsh = m.addMVar(ncbf, name = 'lambda_s_h')
        lph = m.addMVar((ncbf,len(self.bp)), ub = infval, lb = 0, name = 'lambda_p_h')


        lxh = m.addMVar((ncbf,len(self.bx)), lb= 0, ub = infval, name='lambda_x_h')
        rhoh = m.addMVar((ncbf,4,szp), lb= 0, ub = infval, name='rho_h')


        btxh = m.addMVar((ncbf,szp,len(self.bx)), lb= 0, ub = infval, name='beta_x_h')
        etah = m.addMVar((ncbf,szp,2,2), lb = 0, ub =infval, name ='eta_h')

        ########CBF constraints

        for ih in range(ncbf):
        # ih = 0
            rh = -self.ch*self.bH[ih] -self.AH[ih]@self.B@Kb
            
            Mxh = -self.AH[ih]@self.A-self.ch*self.AH[ih]
            temph = (-np.array([self.AH[ih]])@self.B)
            Mph = np.kron(temph, self.kernel.T) @ K.reshape(-1) 
            
            m.addConstr(-Mxh+lph[ih]@self.Ax2+lxh[ih]@self.Ax+((rhoh[ih,0]-rhoh[ih,1])@np.ones(szp)*ex)[0]+((rhoh[ih,2]-rhoh[ih,3])@np.ones(szp)*ey)[0] == 0)
            m.addConstr(rhoh[ih,0]+rhoh[ih,1]==0)
            m.addConstr(rhoh[ih,2]+rhoh[ih,3]==0)
            

            m.addConstr(-(lxh[ih]@self.bx)[0]+(rhoh[ih,0]-rhoh[ih,1])@(self.txT-self.l[0,0]*np.ones((1,szp)))[0]+(rhoh[ih,2]-rhoh[ih,3])@(self.tyT-self.l[1,0]*np.ones((1,szp)))[0]-lph[ih]@self.bp+lsh[ih]+self.sigma_max*lzh[ih][0]+self.sigma_max*lzh[ih][1]+rh<= dh)
            for ix in range(szp):
                m.addConstr(-btxh[ih][ix]@self.bx+(etah[ih][ix][0,0]-etah[ih][ix][0,1])*(-self.txT[0,ix]+self.l[0,0])+(etah[ih][ix][1,0]-etah[ih][ix][1,1])*(-self.tyT[0,ix]+self.l[1,0])<= -Mph[ix]+(lph[ih]@self.Ap)[ix]+lsh[ih])
                m.addConstr(btxh[ih][ix]@self.Ax+((etah[ih][ix][0,0]-etah[ih][ix][0,1])*ex)[0]+((etah[ih][ix][1,0]-etah[ih][ix][1,1])*ey)[0] ==0)
                m.addConstr(lzh[ih][0]-etah[ih][ix][0,0]-etah[ih][ix][0,1]==0)
                m.addConstr(lzh[ih][1]-etah[ih][ix][1,0]-etah[ih][ix][1,1]==0)


            m.update()
            m.setObjective(wv*dv+wh*dh,  GRB.MINIMIZE)
            m.update()
            # m.params.NonConvex = 2
            m.optimize()
        
        if m.Status == GRB.OPTIMAL:
            print('K=', K.X)
            self.K = K.X

            # self.Mp =kernel_sum.X
        
            # self.Mp =0
            # for i in range(len(kernel_ls)):
            #     self.Mp += K[i].X@kernel_ls[i]
            # # print('control gain', K.X)
            print('Kb', Kb.X)
            print('dv,dh =' ,dv.X,dh.X)
            self.Kb = Kb.X
            np.save(os.path.join(self.directory_save, 'K.npy'), self.K)
            np.save(os.path.join(self.directory_save, 'Kb.npy'), self.Kb)


        
class cell ():
    def __init__(self,Barrier, exit_Vertices,vrt ):
        self.bar = Barrier
        # self.wrd = world
        self.exit_vrt = exit_Vertices
        self.vrt = np.array(vrt)
        



    def check_in_polygon(self, p):
            """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
            p = np.reshape(p,(1,2))
            from scipy.spatial import Delaunay
            if not isinstance(self.vrt,Delaunay):
                hull = Delaunay(self.vrt)

            return (hull.find_simplex(p)>=0)[0]
   


def gen_controller_all_orinetation(cell_i, directory_mat, directory_save, ch , cv, eps , sigma_max, dt, measurement_mode ):

    # dt = 0.001
    # # rate_maps = load_rate_map()
    # ch_ls = [100,1,1,10,1,1,100,10,1,100,1,100]
    # cv_ls = [0.1,1,1,1,1,1,0.1,0.1,1,0.9,1,1]

    # cell_ls = [c0]

    # sigma_max_ls = [0.01, 2, 0.1, 2, 2, 2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    # eps_ls = [0.01, 2, 1.5, 2.5, 2, 2, 1, 2, 2, 3, 2, 2]
    # # for i_cell in range(len(cell_ls)):

    # # i_cell = 0

    # # rate_maps_cell = rate_maps_select_triangles(i_cell,rate_maps)
    # # plt.imshow(np.sum(rate_maps_cell, axis= 0))
    # # plt.colorbar()
    # # plt.show()
    for deg in range(0, 360, 60):
            # A = np.zeros((2,2))
        A = np.zeros((2,2))
        # A = np.ones((2,2))*0.1
        B = np.eye((2))
        print('*************************deg=',deg,'*************************')
        s0=Control_cal(cell_i,A, B,dt,ch=ch,cv=cv ,sigma_max= sigma_max,eps=eps , grid_size_x=4,
                        grid_size_y=4, directory_mat =directory_mat+str(deg) , directory_save = directory_save+str(deg), measurement_mode = measurement_mode )
        # print('***************************************cell=',i_cell )
        # print(cv_ls[i_cell], ch_ls[i_cell], sigma_max_ls[i_cell], eps_ls[i_cell])
        # s0.plot_cell()
        # plt.show()
        # #

       
        s0.get_K()
        # s0.u_postion(0.0,0.0)
        

        # s0.check_Probability_constraints(0,0)
        s0.vector_F()
    # np.save('Mp'+str(i_cell), s0.Mp)
    # np.save('Kb'+str(i_cell), s0.Kb)



# delta_x = 0.001




if __name__ == '__main__':
    
    A = np.zeros((2,2))
    B = np.eye((2))
    measurement_mode = 'neural_lidar'
    i_cell = 4
    directory_mat = 'cells_kernels/c'+str(i_cell)+'/deg'
    # directory_save =  'cells_controllers/c'+str(i_cell)+'/deg'
    if measurement_mode == 'vae':
        directory_save = 'cells_controllers_vae'
    elif measurement_mode == 'neural_lidar':
        directory_save = 'cells_controllers'
    elif measurement_mode == 'neural_rate':
        directory_save = 'cells_controllers'
    directory_save = directory_save+'/c'+str(i_cell)+'/deg'
    print("###############################cell", str(i_cell))
    ###Working parameters
    gen_controller_all_orinetation(cell_ls[i_cell], directory_mat, directory_save, ch =1*10**-1, cv=1*10**-4, eps = 10**-2, sigma_max = 10**-6, dt = 0.001, measurement_mode = measurement_mode )
    # gen_controller_all_orinetation(cell_ls[i_cell], directory_mat, directory_save, ch= 5*10**5, cv=1*10**-2, eps = 10**-1, sigma_max = 10**-6, dt = 0.001, measurement_mode = measurement_mode )