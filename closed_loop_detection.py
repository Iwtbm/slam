import numpy as np
import scipy
from skimage.measure import block_reduce
from scipy.sparse import coo_matrix
import heapq
from collections import deque
import matplotlib.pyplot as plt
import time
import scipy.ndimage
from scipy import interpolate
from scipy import optimize
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import multi_dot
from scipy.sparse import coo_matrix
from scipy import ndimage
        


class Grid_process:
    def __init__(self, scale, depth, Origin_grid, Target_grid, Target_tf):                          # Origin_grid: m*n, Target_grid: m*n
        self.score_threshold = 50
        self.scale = scale
        self.depth = depth
        self.Origin_grid = Origin_grid                   # matrix
        self.Target_grid = Target_grid                   # matrix 3 * n
        self.Target_tf = Target_tf
        self.W_x = 7
        self.W_y = 7
        self.W_theta = 30/180 * np.pi                             # search window rotation
        self.d_max = 50
        self.init_split()
        self.resolution_devide()
        self.root_node_create()
        self.Match = self.branch_and_bound_search()


    def init_split(self):
        '''
        calculate the bound of the searching windows
        '''
        self.w_x = np.ceil(self.W_x/self.scale)
        self.w_y = np.ceil(self.W_y/self.scale)
        self.theta_stepsize = np.arccos(1-self.scale**2/2/self.d_max**2)
        self.w_theta = np.ceil(self.W_theta/self.theta_stepsize)
        

    def root_node_create(self):
        '''
        create the original root node in order to branch and bound
        '''
        size_x = np.ceil(2 * self.w_x / 2**self.depth)
        size_y = np.ceil(2 * self.w_y / 2**self.depth)
        self.root = [Node(self.info, self.depth, -self.w_x + i * 2**self.depth, -self.w_y + j * 2**self.depth, -self.W_theta + k * self.theta_stepsize, self.Target_tf) for i in range(int(size_x)) for j in range(int(size_y)) for k in range(int(2*self.w_theta))]    

    
    def resolution_devide(self):
        '''
        create a dictionary contains information of data with difference depth 
        sequence  = origin matrix, origin compressed, target matrix, target compressed  

        '''
        
        self.info = {}
        
        for i in range(self.depth+1):   
            Filter_size = 2**i
            New_origin = ndimage.maximum_filter(self.Origin_grid, size=2*Filter_size-1, mode='constant')
            
            self.info[i] = (New_origin, self.Target_grid)
        
        
    def matrix_compress(self, Matrix):
        '''
        return a 3xn matrix. First row represents Row index,Second row represents column and third Row represent value 
        Input = ndarray (m x n)
        Output = ndarray (3 x k)
        '''
        
        index = np.array(np.nonzero(Matrix))
        value = Matrix[np.nonzero(Matrix)]
        L = np.vstack((index,value))
        return L

    
    def branch_and_bound_search(self):
        Best_score = self.score_threshold
        Stack_sorted = sorted(self.root)
        Stack = deque(Stack_sorted)

        while len(Stack) !=0 :
            
            c = Stack.pop()
            
            if c.score > Best_score:
                if c.c_h == 0:
                    Match = c
                    Best_score = c.score
                else:
                    List = c.expand()
                    List_sorted = deque(sorted(List))
                    Stack = Stack + List_sorted
           
        return Match

        
class Node:
    def __init__(self, info, depth, c_x, c_y, c_theta, Target_tf, parent = None):      # grid 3*N

        self.c_h = depth
        self.c_x = c_x
        self.c_y = c_y                                                  
        self.c_theta = c_theta
        self.Target_tf = Target_tf
        self.info = info
        self.Origin_grid = info[depth][0]
        self.Target_com = info[depth][1]
        self.score_cal()
        self.parent = parent
        self.children = []

    def __eq__(self, s):
        return self.score == s.score

    def __ne__(self, s):
        return self.score != s.score

    def __lt__(self, s):
        return self.score < s.score

    def __le__(self, s):
        return self.score <= s.score

    def __gt__(self, s):
        return self.score > s.score

    def __ge__(self, s):
        return self.score >= s.score

    def get_score(self):
        return self.score

    def Grid_Mapping(self, Matrix):
        points = np.ceil(Matrix[:2, :]).astype(int)

        return points
         
    def score_cal(self):
        translation = np.array([[self.c_x, self.c_y, 1]])
        rotation = np.array([[np.cos(self.c_theta), - np.sin(self.c_theta)], [np.sin(self.c_theta), np.cos(self.c_theta)], [0, 0]])
        target_points = np.dot(np.dot(np.hstack((rotation, translation.T)),self.Target_tf), np.vstack((self.Target_com[:2, :], np.ones(self.Target_com.shape[1]))))
        grid = self.Grid_Mapping(target_points)        # 3 * n
        self.score = sum(self.Origin_grid[(grid[0], grid[1])])
    
        
    def expand(self):
        if len(self.children) == 0:                # use c_x, c_y many times
            child = [[self.c_x, self.c_y], [self.c_x, self.c_y + 2**(self.c_h-1)], [self.c_x + 2**(self.c_h-1), self.c_y], [self.c_x + 2**(self.c_h-1), self.c_y + 2**(self.c_h-1)]] 
            for i in range(4):
                self.children.append(Node(self.info, self.c_h - 1, child[i][0], child[i][1], self.c_theta, self.Target_tf, parent = self))
        return self.children
        
    
def matrix_decompress(Data,M = 1001,N =1001,origin_x = 500, origin_y = 500):
    
    m,n = Data.shape
    Data[0,:] += origin_x
    Data[1,:] += origin_y
    if m == 2 :
        Data = np.vstack((Data,np.ones([1,n])))
        M = coo_matrix((Data[2,:],(Data[0,:],Data[1,:])), shape=(M,N)).toarray()
        return M
    elif m == 3:
        M = coo_matrix((Data[2,:],(Data[0,:],Data[1,:])), shape=(M,N)).toarray()
        return M
    else:
        return "matrix cannot decompress"