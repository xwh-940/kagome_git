from statistics import multimode
from code_func import *

#multi = 2 back
multi = 2
kn = para.kn
Tn = para.Tn
N = Tn*kn*kn*kn
T = para.T

self_consistent_number = para.self_consistent_number
iteration_number = para.iteration_number

U0 = para.u0 
U1 = U0
J = U0 * 0.1 
J1 = J 
U11 = U0 - 2 * J


k_orgin = np.linspace(-1,1,kn+1)*multi*(1/sqrt(3))
k_orgin_1 = (k_orgin[1:]) * np.pi
#k_orgin_1 = (k_orgin[1:]+k_orgin[:-1]) * np.pi / 2
wn = np.arange(-Tn+1, Tn+1, 2)*np.pi*T
dirac_e = wn[int(Tn/2)]*0
#print_and_save_txt("dirac_e:",dirac_e)
print_and_save_txt("dirac_e",dirac_e)

kx_band = ky_band = k_orgin_1
kx_band, ky_band = np.meshgrid(kx_band,ky_band)

kx_band_fs = ky_band_fs = (np.linspace(-1,1,para.kn_fs+1)*multi*(1/sqrt(3)))[1:]*np.pi
kx_band_fs, ky_band_fs = np.meshgrid(kx_band_fs,ky_band_fs)
kagome_lattice=[[2/3*multi, 0], [1/3*multi, 1/sqrt(3)*multi], [-1/3*multi,1/sqrt(3)*multi], [-2/3*multi,0],[-1/3*multi,-1/sqrt(3)*multi],[1/3*multi,-1/sqrt(3)*multi]]
kagome_lattice_bz=[[1/3*multi, 0], [1/6*multi, 1/sqrt(3)*multi/2], [-1/3*multi/2,1/sqrt(3)*multi/2], [-1/3*multi,0],[-1/6*multi,-1/sqrt(3)*multi/2],[1/6*multi,-1/sqrt(3)*multi/2]]


