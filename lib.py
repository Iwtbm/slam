#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 15:33:45 2018

@author: wendaxu, yifengluo
"""
import numpy as np
import time
import scipy.ndimage
import heapq
from scipy import interpolate
from scipy import optimize
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import multi_dot
from scipy.sparse import coo_matrix
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
            Q = self.Dict[i] #change the sequence,so change to transform matrix P_Q
            P = self.Dict[i+1]
            down = int(P.shape[1] * ratio[0])
            up = int(P.shape[1] * ratio[1])
            self.neigh = NearestNeighbors(n_neighbors=1)
            self.neigh.fit(Q[0:2,:].T)
            self.P_now = np.copy(P[:,down:up])
            self.P_now = np.vstack((self.P_now,np.ones([1,self.P_now.shape[1]])))
            result  = optimize.minimize(self.least_square,initial)
            t_x,t_y,theta = result.x
        
            T =  np.array([[np.cos(theta/180.0*np.pi),-np.sin(theta/180.0*np.pi),t_x],[np.sin(theta/180.0*np.pi),np.cos(theta/180.0*np.pi),t_y],[0,0,1]])
            
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


    def Map_create(self):
        self.Transform_P2O = {}
        Max = max(self.Dict)
        Concantenate = min(self.Dict)
        Min = Concantenate + 1
	
        for i in range(Min,Max):
            Matrix_list = [self.Transform_P2Q[j] for j in range(Min,i+1)]
            #Matrix_list.reverse()
            if len(Matrix_list)>1:	
                Matrix = multi_dot(Matrix_list)                                         # Test function
            else:
                Matrix = Matrix_list[0]
                self.Transform_P2O[i+1] = Matrix
        self.relation_matrix = np.dot(self.Transform_P2Q[Concantenate],self.parent)     # which should be stored
        self.relation_matrix_to_next = self.Transform_P2O[Max]                          # which should be transferred to next icp class
        self.min = Min
        self.max = Max

        

