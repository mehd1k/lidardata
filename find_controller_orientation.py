import os
import numpy as np
from cell_config import cell, cell_ls
class control_gain_load():
    def __init__(self):
       
        self.dir_control_gains = 'cells_controllers'
       


    def load_matrix_from_folder(self,folder_path):
        """
        Load the matrix from the given folder.
        
        Args:
            folder_path (str): Path to the folder containing the matrix.
        
        Returns:
            np.ndarray: The matrix loaded from the folder.
        """
        Kb_path = os.path.join(folder_path, 'Kb.npy')  
        K_path = os.path.join(folder_path, 'K.npy')  
        if os.path.exists(Kb_path) and os.path.exists(K_path):
            return np.load(K_path) , np.load(Kb_path)  
        else:
            raise FileNotFoundError(f"No matrix file found in {folder_path}")

    def interpolate_matrices(self, matrix1, matrix2, degree1, degree2, target_degree):
        """
        Interpolate element-wise between two matrices based on the target orientation.
        
        Args:
            matrix1 (np.ndarray): The matrix corresponding to degree1.
            matrix2 (np.ndarray): The matrix corresponding to degree2.
            degree1 (float): The degree associated with matrix1.
            degree2 (float): The degree associated with matrix2.
            target_degree (float): The target orientation for interpolation.
        
        Returns:
            np.ndarray: The interpolated matrix.
        """
        # Compute the interpolation factor (linear interpolation formula)
        factor = (target_degree - degree1) / (degree2 - degree1)
        
        # Interpolate element-wise between the two matrices
        interpolated_matrix = (1 - factor) * matrix1 + factor * matrix2
        
        return interpolated_matrix


        



    def interpolate_contorlgains(self, cell_number, orientation):
        """
        Interpolate the matrix for a given orientation by loading only the two closest matrices.
        
        Args:
            base_folder (str): The path to the main folder containing 'deg' subfolders.
            orientation (float): The orientation angle for which the matrix is to be interpolated.
        
        Returns:
            np.ndarray: The interpolated matrix.
        """
        # Find the two closest degree
        # Normalize orientation to [0, 360) range
        orientation = orientation % 360
        
        orientation_list_folder = list(np.arange(0, 360, 90))
    
        # Find the two orientations that bracket the target orientation
        lower_degree = None
        upper_degree = None
        
        # Check if orientation exactly matches a value in the list
        if orientation in orientation_list_folder:
            idx = orientation_list_folder.index(orientation)
            # Use the exact match and the next value
            lower_degree = orientation_list_folder[idx]
            if idx < len(orientation_list_folder) - 1:
                upper_degree = orientation_list_folder[idx + 1]
            else:
                # Wrap around: last value wraps to 0 (360)
                upper_degree = 360
        else:
            # Find the two values that bracket the orientation
            for i in range(len(orientation_list_folder)):
                if orientation_list_folder[i] > orientation:
                    # Found the upper bound
                    upper_degree = orientation_list_folder[i]
                    # Lower bound is the previous element
                    if i > 0:
                        lower_degree = orientation_list_folder[i - 1]
                    else:
                        # Wrap around: lower is the last element (270), upper is 0 (which becomes 360)
                        lower_degree = orientation_list_folder[-1]
                        upper_degree = 360  # For interpolation, treat 0 as 360
                    break
            
            # Handle case where orientation is >= last value in list (wrap-around case)
            if lower_degree is None or upper_degree is None:
                # Orientation is between last value and 360 (which wraps to 0)
                lower_degree = orientation_list_folder[-1]  # 270
                upper_degree = 360  # For interpolation purposes
        
        # Load matrices from the two closest folders
        # Handle wrap-around: if upper_degree is 360, use deg0 folder
        upper_degree_for_folder = 0 if upper_degree == 360 else upper_degree
        lower_folder = os.path.join(self.dir_control_gains,'c'+str(cell_number),f'deg{lower_degree}')
        upper_folder = os.path.join(self.dir_control_gains,'c'+str(cell_number), f'deg{upper_degree_for_folder}')
        
        K1,Kb1 = self.load_matrix_from_folder(lower_folder)
        K2,Kb2 = self.load_matrix_from_folder(upper_folder)
        
        # Perform interpolation
        if lower_degree == upper_degree:
            return K1, Kb1
        else:
            K_interpolated = self.interpolate_matrices(K1, K2, lower_degree, upper_degree, orientation)
            Kb_interpolated = self.interpolate_matrices(Kb1, Kb2, lower_degree, upper_degree, orientation)
            
            return K_interpolated, Kb_interpolated
        








if __name__ == '__main__':
    # Example usage:
    base_folder = '/path/to/robot/orientations'  # Replace with the actual path
    orientation = 45  # Example: interpolate matrix for 45 degrees

    cl = control_gain_load()
    folder_path = 'cells_controllers\c0\deg80'
    K, Kb = cl.load_matrix_from_folder(folder_path)
    print('K=', K)
    print('Kb=', Kb)
    # print(f"Interpolated matrix for orientation {orientation}°:\n", K,Kb)
