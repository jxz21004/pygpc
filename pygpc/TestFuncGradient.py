"""
Chenyifu 
=============================
Gradient of test functions
"""
import os
import numpy as np
from .RandomParameter import *
from collections import OrderedDict


class TestFuncGradient:

    def __init__(self):
        pass


class Ishigami_Grad(TestFuncGradient):

    """
    caculate the analytical gradient of Ishigami testfunction

    Parameters
    ----------
    p : 
        Dictionary of model contained the information of parameters value
    parameters :
        Dictionary of random parameters
    GradientDict :
        Dictionary of model gradient
    Gradient : ndarray of float [n_grid * n_out * dim]
        Gradient array
    """
    def __init__(self,parameters):
        
        self.p = None
        self.parameters = parameters

        self.GradientDict = None
        self.Gradient = None


    def cal_gradient(self,p,ngrid,dim,nout=1):

        """
            Caculate the gradient of each grid point

        """
        self.p = p

        # determined the gradient 
        #   d    x
        #   ------ = std
        #   d kesi
        #   different gradient caculation for different type of random parameter 
        coef = OrderedDict()
        for i,key in enumerate(self.parameters):

            if(isinstance(self.parameters[key],RandomParameter)):

                if(isinstance(self.parameters[key],Beta)):
                    coef[key] = 0.5 * (self.parameters[key].pdf_limits[1] - self.parameters[key].pdf_limits[0])
                elif(isinstance(self.parameters[key],Norm)):
                    coef[key] = self.parameters[key].std
                elif(isinstance(self.parameters[key],Gamma)):
                    coef[key] = self.parameters[key].pdf_shape[1]
            else:
                coef[key] = 0

        # Initial gradient dict
        self.GradientDict = OrderedDict()
        for i,key in enumerate(self.p):
            if(i<3):
                self.GradientDict[key] = coef[key]

        if self.p["x1"] is not np.ndarray:
            self.p["x1"] = np.array(self.p["x1"])

        if self.p["x2"] is not np.ndarray:
            self.p["x2"] = np.array(self.p["x2"])

        if self.p["x3"] is not np.ndarray:
            self.p["x3"] = np.array(self.p["x3"])

        if self.p["a"] is not np.ndarray:
            self.p["a"] = np.array(self.p["a"])

        if self.p["b"] is not np.ndarray:
            self.p["b"] = np.array(self.p["b"])

        # caculate gradient of Ishigami function
        self.GradientDict["x1"] = self.GradientDict["x1"] * (np.cos(self.p["x1"].flatten()) + self.p["b"].flatten() * self.p["x3"].flatten() ** 4 * np.cos(self.p["x1"].flatten()))

        self.GradientDict["x2"] = self.GradientDict["x2"] * (2 * self.p["a"].flatten() * np.sin(self.p["x2"].flatten()) * np.cos(self.p["x2"].flatten()))

        self.GradientDict["x3"] = self.GradientDict["x3"] * (4 * self.p["b"].flatten() * self.p["x3"].flatten() ** 3 * np.sin(self.p["x1"].flatten()))
        
        # convert Gradient value from dictionary to array
        self.Gradient = np.zeros((ngrid,nout,dim))
        for i in range(ngrid) :
            for j in range(nout):
                for k in range(dim):
                    for index,key in enumerate(self.GradientDict):
                        if k == index:
                            self.Gradient[i,j,k] = self.GradientDict[key][i]
                            
        return self.Gradient



