#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 15:23:45 2018

@author: wendaxu
"""
import numpy as np
import time
import scipy.ndimage
import heapq
from scipy import interpolate
from scipy import optimize
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import multi_dot
class Data_process:
    def __init__(self,data):
        self.data = np.array(data)
        self.Len = len(self.data)
        self.valid_len = 0 
        
    def valid_Data(self):

        self.processed_data = np.zeros([2,np.sum(self.data != 60)])
        num = 0
        for i in range(self.Len):
            if self.data[i] != 60:
                self.processed_data[0,num] = i
                self.processed_data[1,num] = self.data[i]
                num+=1
        self.valid_len = self.processed_data.shape[1]
        return self.processed_data
    def axis_change(self):
        if self.processed_data.shape[0] != 2:
            print("Can't process this data")
            return 0
        else:
            if self.valid_len == 0 :
                print("Can't process this data")
                return 0
            New_Data = np.zeros(self.processed_data.shape)
            for i in range(self.valid_len):
                Degree = self.processed_data[0,i]
                Range = self.processed_data[1,i]
                New_Data[0,i] = np.cos(Degree/self.Len * np.pi) * Range
                New_Data[1,i] = np.sin(Degree/self.Len * np.pi) * Range
            self.processed_data = New_Data
            
    def add_dimension(self):
        self.processed_data = np.vstack((self.processed_data,np.ones([1,self.processed_data.shape[1]])))
        return self.processed_data
    

class Grid_process:
    def __init__(self,data,scale,depth, Origin_grid, Target_grid):
        self.data = data
        self.scale = scale
        self.depth = depth
        self.Origin_grid = Origin_grid
        self.Target_grid = Target_grid
        self.W_x = 7
        self.W_y = 7
        self.W_theta = 30
        self.init_split()

        
    def init_split(self):
        '''
        calculate the bound of the searching windows
        '''
        self.w_x = np.ceil(self.W_x/self.scale)
        self.w_y = np.ceil(self.W_y/self.scale)
        self.w_theta = np.ceil(self.W_theta/(1-scale**2/2/self.d_max**2))
        

    def root_node_create(self):
        '''
        create the original root node in order to branch and bound
        '''
        size_x = np.ceil(2 * self.w_x / 2**self.depth)
        size_y = np.ceil(2 * self.w_y / 2**self.depth)
        self.root = [Node(self.info, depth, - self.w_x + i * 2**self.depth, -self.w_y + j * 2**self.depth, self.w_theta * np.arccos((1 - scale**2 / 2 / self.d_max**2))) for i in range(size_x) for j in range(size_y) for k in range(self.w_theta) ]    
    
    
    def resolution_devide(self):
        '''
        create a dictionary contains information of data with difference depth 
        sequence  = origin matrix, origin compressed, target matrix, target compressed  

        '''
        
        self.info = {}
        for i in range(self.depth):
            Filter_size = 2**i
            New_origin = self.max_pool(Matrix = self.Origin_grid, filter_size = Filter_size)
            Com_origin = self.matrix_compress(New_origin)
            New_target = self.max_pool(Matrix = self.Target_grid, filter_size = Filter_size)
            Com_target = self.matrix_compress(New_target)
            self.info[i] = (New_origin,Com_origin,New_target,Com_target)
            
            
    def max_pool(self,Matrix,filter_size = 1):
        '''
        change the resolution
        Input = ndarray (m x n), Filter (i x i)
        Output = ndarray (m x n)
        '''
        if filter_size == 1:
            return Matrix        
        New = scipy.ndimage.filters.maximum_filter(Matrix,size = filter_size)
        return New
        
        
    def matrix_compress(self,Matrix):
        '''
        return a 3xn matrix. First row represents Row index,Second row represents column and third Row represent value 
        Input = ndarray (m x n)
        Output = ndarray (3 x k)
        '''
        
        index = np.nonzero(Matrix)
        value = Matrix[index]
        L= np.vstack((np.vstack((index[0],index[1])),value))
        return L

    
    def branch_and_bound_search(self):
        Best_score = self.score_threshold
        Stack = self.root
        heapq.heapify(Stack)
        while len(Stack) !=0 :
            c = heapq.heappop(Stack)
            if c.score > Best_score:
                if c.c_h == 0:
                    Match = c
                    Best_score = c.score
                else:
                    List = c.expand()
                    for cc in List:
                        heapq.heappush(Stack,cc)
        return Best_score,Match
        
        
        
        
        
        
class Node:
    def __init__(self,info, depth, c_x, c_y, c_theta, parent):      # grid 3*N
        self.c_h = depth
        self.c_x = c_x
        self.c_y = c_y                                                                                                       # resolution
        self.c_theta = c_theta
        self.info = info
        self.Origin_grid = info[depth][0]
        self.Target_grid = info[depth][2]
        self.Origin_com = info[depth][1]
        self.Target_com = info[depth][3]
        self.score_cal()
        self.parent = None
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
         
    def score_cal(self):
        # maps = Origin_grid.nonzero()
        maps = self.Origin_com
        KD_trees = scipy.spatial.KDTree(np.array(list(zip(maps[0, :], maps[1, :]))))
        translation = np.array([self.c_x, self.c_y, 1])
        rotation = np.array([np.cos(self.c_theta), - np.sin(self.c_theta)], [np.sin(self.c_theta), np.cos(self.c_theta)], [0, 0])
        grid = np.dot(np.hstack((rotation, translation.T)), np.vstack((self.Target_com[:2, :], np.ones(self.Target_com.shape[1]))))
        pts = grid
        pts = np.array(list(zip(pts[0, :], pts[1, :])))
        dist_list = KD_trees.query(pts)
        self.score = np.mean(np.power(dist_list[0], 2))
        
    def expand(self):
        if len(self.children) == 0:
	    
            child = [[self.c_x, self.c_y], [self.c_x, self.c_y + 2**(self.c_h-1)], [self.c_x + 2**(self.c_h-1), self.c_y], [self.c_x + 2**(self.c_h-1), self.c_y + 2**(self.c_h-1)]] 
            for i in range(4):
                self.children.append(Node(self.info, self.c_h - 1, child[i][0], child[i][1], self.c_theta, parent = self))
        return self.children
    
            
                
    
                
    
    
class ICP:
    '''
    Transform_P2Q is matrix of near pic, ie Transform_P2Q[12] is pic_13 to pic_12
    Transform_P2O is matrix of origin and target, ie Transform_P2O[12] is pic_12 to pic_9(
    assume pic_9 is the orginal pic)

    '''
    def __init__(self,Dict = {},parent = np.eye(3)):
        self.Dict = Dict
        self.Len = len(self.Dict)
        self.Transform_P2Q = {}
        self.resolution = 10
        self.worldmap_size = [1001, 1001] 
        self.size = [1001, 1001]
        self.origin = [500, 500]
        self.lo_occ = 1
        self.lo_free = 0.5
        self.lo_max = 100
        self.lo_min = -100
        self.worldmap = np.zeros(self.worldmap_size)

        self.relation_matrix = np.eye(3)
        self.parent = parent
    
    def update(self):
        Len_Trans = len(self.Transform_P2Q)
        Len_Dict = self.Len
        if Len_Trans == Len_Dict-1 :
            return "No transform matrix updated"
        elif Len_Trans > Len_Dict-1 :
            self.clean_transform()
            return "Error, clean transform matrix"
        else:
            number = Len_Dict - Len_Trans -1 
            Dict_new = dict([(Len_Trans+i,self.Dict[Len_Trans+i] ) for i in range(number+1)])
            self.fit(Target = Dict_new )
            
        
        
    def fit(self,ratio = [0.2,0.8],initial = [0,0,0],Target = 0):
        if Target ==0 :
            Target = self.Dict
            i = min(Target)
        while 1:
            print("Matching %dth pic"%(i))
            a = time.clock()
            Q = self.Dict[i] #change the sequence,so change to transform matrix P_Q
            P = self.Dict[i+1]
            down = int(P.shape[1] * ratio[0])
            up = int(P.shape[1] * ratio[1])
            print("1th = ", time.clock() - a)
            a = time.clock()
            self.neigh = NearestNeighbors(n_neighbors=1)
            self.neigh.fit(Q[0:2,:].T)
            #self.KD_trees = scipy.spatial.KDTree(Q[0:2,:].T)
            print("2th = ", time.clock() - a)
            a = time.clock()
            self.P_now = np.copy(P[:,down:up])
            self.P_now = np.vstack((self.P_now,np.ones([1,self.P_now.shape[1]])))
 
            result  = optimize.minimize(self.least_square,initial)
            print("3th = ", time.clock() - a)
            a = time.clock()
            t_x,t_y,theta = result.x
        
            T =  np.array([[np.cos(theta/180.0*np.pi),-np.sin(theta/180.0*np.pi),t_x],[np.sin(theta/180.0*np.pi),np.cos(theta/180.0*np.pi),t_y],[0,0,1]])
            print("4th = ", time.clock() - a)
            
            self.Transform_P2Q[i] = T
            
            if i+1 == max(Target) :
                break
                i += 1
    def clean_transform(self):
        self.Transform_P2Q = {}
        return "clean successful"
    def clean_data(self):
        self.Data = {}
        self.Transform_P2Q = {}
        return "clean successful"
    def add_data(self,data):
        if isinstance(data,dict):
            Len = len(data)
            for i in range(Len):
               self.Dict[self.Len + i] = data[i]
            self.Len += Len
        
        elif isinstance(data,np.ndarray):
            
            self.Dict[self.Len] = data
            self.Len += 1
        else:
            print("invalid data type")
        
    def least_square(self,x):
        Len = self.P_now.shape[1]
        Loss = 0
        t_x,t_y,theta = x.tolist()
        Transform_matrix =  np.array([[np.cos(theta/180.0*np.pi),-np.sin(theta/180.0*np.pi),t_x],[np.sin(theta/180.0*np.pi),np.cos(theta/180.0*np.pi),t_y],[0,0,1]])
        P_after = np.dot(Transform_matrix,self.P_now)
        target = P_after[0:2,:]
        #result = self.KD_trees.query(target.T,1)
        #Loss = np.sum(result[:][0]**2)
        distances, indices = self.neigh.kneighbors(target.T, return_distance=True)
        Loss = np.sum(distances**2)
        Loss = Loss/ Len
        return Loss

#Time = time.clock()
#index = np.nonzero(matrix)
#value = matrix[index]
#L= np.vstack((np.vstack((index[0],index[1])),value))
#print(time.clock() - Time)
#
#
#i = ICP(Dict = Data1)
    #def interpolation(self, submap_matrix, scan_matrix):  # how to optimize scan to submap
    def Occupied_Grid_Mapping(self, Matrix, pose = np.eye((3,3))): # pose:transform matrix
        if pose.all() == np.zeros.all():
            current_map = np.zeros(self.size)
        else:
            current_map = self.worldmap
        x_robot = pose[0][2]
        y_robot = pose[1][2]
        grid_robot = np.ceil(np.array([self.resolution * x_robot, self.resolution * y_robot])).astype(int)
        Matrix = np.vstack((Matrix, np.ones(Matrix.shape[1])))
        points = np.dot(pose, Matrix)
        for i in range(points.shape[1]):
            x_point = points[0,i]
            y_point = -points[1,i]  # or change the minus to plus
            grid_point = (np.ceil((x_point * self.resolution)).astype(int), np.ceil((y_point * self.resolution)).astype(int))
            grid_point = np.array(grid_point).reshape(2) 
            free_points = np.array(self.get_line(grid_robot, grid_point))
            if free_points.any():
                current_map[free_points[:, 1] + self.origin[1], free_points[:, 0] + self.origin[0]] -= self.lo_free
            else:
                current_map[self.origin[1], self.origin[0]] -= self.lo_free
            current_map[self.origin[1]+x_point, self.origin[0]+y_point] += self.lo_occ
 
        current_map[current_map > self.lo_max] = self.lo_max
        current_map[current_map < self.lo_min] = self.lo_min 
        
        return current_map
    

    def get_line(self, start, end):
        """Bresenham's Line Algorithm
        Produces a list of tuples from start and end
    
        >>> points1 = get_line((0, 0), (3, 4))
        >>> points2 = get_line((3, 4), (0, 0))
        >>> assert(set(points1) == set(points2))
        >>> print points1
        [(0, 0), (1, 1), (1, 2), (2, 3), (3, 4)]
        >>> print points2
        [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
        """
        # Setup initial conditions
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
    
        # Determine how steep the line is
        is_steep = abs(dy) > abs(dx)
    
        # Rotate line
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
    
        # Swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True
    
        # Recalculate differentials
        dx = x2 - x1
        dy = y2 - y1
    
        # Calculate error
        error = int(dx / 2.0)
        ystep = 1 if y1 < y2 else -1
    
        # Iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx
    
        # Reverse the list if the coordinates were swapped
        if swapped:
            points.reverse()
        
        return points    

    def Map_create(self):
        self.Transform_P2O = {}
        Max = max(self.Dict)
        Concantenate = min(self.Dict)
        Min = Concantenate + 1

        for i in range(Min,Max):
            Matrix_list = [self.Transform_P2Q[j] for j in range(Min,i+1)]
            Matrix_list.reverse()
            Matrix = multi_dot(Matrix_list)  # Test function
            self.Transform_P2O[i+1] = Matrix
        self.relation_matrix_to_next = self.Transform_P2O[Max]#which should be transferred to next icp class
        self.relation_matrix = np.dot(self.Transform_P2Q[Concantenate],self.parent)# which should be stored

        


