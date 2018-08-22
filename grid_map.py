import numpy as np
from matplotlib import pyplot as plt
import time

class grid_map:
    def __init__(self):
        self.resolution = 25
        self.size = [1000, 1000]
        self.origin = [500, 500]
        self.lo_occ = 0.9
        self.lo_free = 0.7
        self.lo_max = 100
        self.lo_min = -100

    def Occupied_Grid_Mapping(self, ranges, angles, pose):
        n = ranges.shape[1]
        current_map = np.zeros(self.size)
        
        for i in range(n):
            start = time.clock()
            x_robot = pose[0, i]
            y_robot = pose[1, i]
            theta_robot = pose[2, i]
            grid_robot = np.ceil(np.array([self.resolution * x_robot, self.resolution * y_robot])).astype(int)
            

            for j in range(len(angles)):
                x_point = ranges[j, i] * np.cos(theta_robot + angles[j]) + x_robot
                y_point = - ranges[j, i] * np.sin(theta_robot + angles[j]) + y_robot
                grid_point = (np.ceil(x_point * self.resolution).astype(int), np.ceil(y_point * self.resolution).astype(int))
                grid_point = np.array(grid_point).reshape(2) 
                free_points = np.array(self.get_line(grid_robot, grid_point))

                if free_points.any():
                    current_map[free_points[:, 1] + self.origin[1], free_points[:, 0] + self.origin[0]] -= self.lo_free
                else:
                    current_map[self.origin[1], self.origin[0]] -= self.lo_free

                current_map[grid_point[1] + self.origin[1], grid_point[0] + self.origin[0]] += self.lo_occ
            print(time.clock() - start)
        current_map[current_map > self.lo_max] = self.lo_max
        current_map[current_map < self.lo_min] = self.lo_min 

        self.build_map(current_map)

    def build_map(self, current_map):
        plt.imshow(current_map, cmap='gray')
        plt.show()




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