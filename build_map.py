#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

class gridmap:
    def __init__(self,Dict = {},parent = np.eye(3)):
        # self.Dict = Dict
        # self.Len = len(self.Dict)
        # self.Transform_P2Q = {}
        self.resolution = 10
        self.worldmap_size = [1001, 1001] 
        self.size = [1001, 1001]
        self.origin = [500, 500]
        self.lo_occ = 0.9
        self.lo_free = 0.7
        self.lo_max = 100
        self.lo_min = -100
        self.worldmap = np.zeros(self.worldmap_size)
        self.tf = np.eye((3,3))
        
        # self.relation_matrix = np.eye(3)
        # self.parent = parent

    def Occupied_Grid_Mapping(self, Matrix, pose): # pose:transform matrix
        n = 5
        for j in range(n):
            self.tf = np.dot(self.tf, pose[j])
            x_robot = self.tf[0][2]
            y_robot = self.tf[1][2]
            grid_robot = np.ceil(np.array([self.resolution * x_robot, self.resolution * y_robot])).astype(int)
            matrix_p = np.vstack((Matrix[j], np.ones(Matrix[j].shape[1])))
            points = np.dot(self.tf, matrix_p)  # or Matrix * tf?
            for i in range(points.shape[1]):
                x_point = points[0,i]
                y_point = -points[1,i]  # or change the minus to plus
                grid_point = (np.ceil((x_point * self.resolution)).astype(int), np.ceil((y_point * self.resolution)).astype(int))
                grid_point = np.array(grid_point).reshape(2)
                free_points = np.array(self.get_line(grid_robot, grid_point))
                if free_points.any():
                    self.worldmap[free_points[:, 1] + self.origin[1], free_points[:, 0] + self.origin[0]] -= self.lo_free
                else:
                    self.worldmap[self.origin[1], self.origin[0]] -= self.lo_free
                self.worldmap[self.origin[1]+x_point, self.origin[0]+y_point] += self.lo_occ
            self.worldmap[self.worldmap > self.lo_max] = self.lo_max
            self.worldmap[self.worldmap < self.lo_min] = self.lo_min
        
        return self.worldmap
    

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