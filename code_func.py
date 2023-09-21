from shapely.geometry import Point, asShape
import numpy as np
import sys, os, time
import itertools as it
import numba
import mkl_fft as fft
#from numpy import fft
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import jit, prange,njit,vectorize
from numpy import ascontiguousarray as con
from numexpr import evaluate as ne
import torch as tr
import para_code as para
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigs
from scipy.optimize import approx_fprime
from scipy.interpolate import RectBivariateSpline
from scipy import optimize

inv = np.linalg.inv
f = np.float32
v15 = f(1.5)
v05 = f(0.5)
fl = float
arg_ = sys.argv
s = str
vec = np.vectorize
eig=np.linalg.eigh
sqrt = np.sqrt
pi = np.pi
norm = np.linalg.norm
cos = np.cos
sin = np.sin

run_time = s(time.strftime("%Y%m%d %Hh%Mmin%Ss", time.localtime()))
name_time = "mu=" + s(para.mu) + "_u0=" + s(para.u0) + "_u1=" + s(para.u1)\
          + "_u11=" + s(para.u11) + "_j=" + s(para.j) + "_j11=" + s(para.j11) \
          + "_" + s(para.kn) + "_" + s(para.Tn) + "_" + s(para.channel) + "_" +  s(para.symm) + "_" + s(para.anti)
if para.test == 0:
    "test"
    SOURCE = "20230222"
    name_time = run_time
if para.test == 1:
    "true compute"
    SOURCE = para.result_name    

data = "data"
path_model_fig = SOURCE+"//"+name_time+"//model_fig"
path_chi0_fig = SOURCE+"//"+name_time+"//chi0_fig"
path_chi0s_fig = SOURCE+"//"+name_time+"//chi0s_fig"
path_chi0r_fig = SOURCE+"//"+name_time+"//chi0r_fig"
path_chis_fig = SOURCE+"//"+name_time+"//chis_fig"
path_phi_fig = SOURCE+"//"+name_time+"//phi_fig"
#path_phi_std0_fig = SOURCE+"//"+name_time+"//phi_fig_std0"
#path_phi_std1_fig = SOURCE+"//"+name_time+"//phi_fig_std1"
#path_fermi_suface_fig = SOURCE+"//"+name_time+"//fermi_suface_fig"

if not os.path.exists(SOURCE):
    os.makedirs(SOURCE)

if not os.path.exists(data):
    os.makedirs(data)

os.makedirs(SOURCE+"//"+name_time)
os.makedirs(path_model_fig)
os.makedirs(path_chi0_fig)
os.makedirs(path_chi0s_fig)
os.makedirs(path_chi0r_fig)
os.makedirs(path_chis_fig)
os.makedirs(path_phi_fig)
#os.makedirs(path_phi_std0_fig)
#os.makedirs(path_phi_std1_fig)
#os.makedirs(path_fermi_suface_fig)


txt_print = SOURCE+"//"+name_time+"//output.txt"
def print_and_save_txt(*arg):
    output_writter = open(txt_print, "a+")
    arg_number = len(arg)
    for i in range(arg_number):
        print(s(arg[i])+" ", end='')
    print()
    for i in range(arg_number):
        output_writter.write(s(arg[i])+" ")
    output_writter.write("\n")
    output_writter.close()

#name time
if para.test == 1:
    "true compute"
    print_and_save_txt(name_time)

#print(numba.config.NUMBA_NUM_THREADS)
#import scipy as sp
numba.set_num_threads(4)
print_and_save_txt("num threads:", numba.get_num_threads())


