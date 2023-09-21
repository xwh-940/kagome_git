from parameters import *
from filename import *
from math_func import *


t_xz = 1
t_yz = 0.5
t_l = 0.00
mu = para.mu
e_xz = 2.182 + mu + 0.1
#e_xz = 1 + mu 
e_yz = -0.055 + mu + 0.2

orbit_number = 2
sublattice_number = 3
band_number = orbit_number * sublattice_number
band_number_square = band_number ** 2





kx = ky = k_orgin_1

#kx_basis = np.array([0, 1])
#ky_basis = np.array([sqrt(3)/2, -1/2])
ky_basis = np.array([sqrt(3)/2, 1/2])
kx_basis = np.array([sqrt(3)/2, -1/2]) # true

para_array = np.array(list(\
             it.product(range(Tn), range(kn), range(kn))))
para_array_kn = np.array(list(\
             it.product(range(kn), range(kn))))
orbitral_array =  np.array(list(it.product(range(band_number),range(band_number))))

@njit
def dirac_func(index0,index1):
    if index0==index1:
        return 1
    else:
        return 0

@njit
def factor(kx, ky, arr1, arr2):
    #k_vector = np.array([kx,ky]).astype(np.complex64)
    k_vector = (kx*kx_basis+ky*ky_basis).astype(np.complex64)
    arr_a = np.array([arr1, arr2]).astype(np.complex64)
    num = np.complex64(2j)
    return 1.0 + np.exp(-num*k_vector@arr_a)
    #return 2*np.cos(np.complex64(1.0)*k_vector@arr_a)


@njit
def Phi(i, j, kx,ky):
    '''
    lattice structure factors
    A, B, C: i = 0, 1, 2
    '''
    if i==j:
        return 0
    if i==1 and j==0:
        #a1
        return factor(kx, ky, 1/2, np.sqrt(3)/2)
    if i==0 and j==2:
        #a3
        return factor(kx, ky, -1, 0)
    if i==1 and j==2:
        #a2
        return factor(kx, ky, -1/2, np.sqrt(3)/2)
    else:
        return np.conj(Phi(j, i, kx, ky))



@vectorize
def hamiltonian_xz(i,j,kx,ky):
    return e_xz*dirac_func(i,j) - t_xz*Phi(j,i,kx,ky)

@vectorize
def hamiltonian_yz(i,j,kx,ky):
    return e_yz*dirac_func(i,j) - t_yz*Phi(j,i,kx,ky)

#hamiltonian_vec = vec(hamiltonian, signature='(),(),(),(),(),()->()', otypes=[complex])
sublattice_arr = con(np.arange(sublattice_number))

num_sublattice = sublattice_arr.shape[0]
num_band = num_sublattice
sublattice_arr1 = sublattice_arr.reshape(num_sublattice,1)
sublattice_arr2 = sublattice_arr.reshape(1,num_sublattice)

@njit
def hamiltonian_matrix_xz(kx,ky):
    matrix=hamiltonian_xz(sublattice_arr1, sublattice_arr2, kx, ky)
    matrix_trans = con(matrix)
    return con(matrix_trans).astype(np.complex64)

@njit
def hamiltonian_matrix_yz(kx,ky):
    matrix=hamiltonian_yz(sublattice_arr1, sublattice_arr2, kx, ky)
    matrix_trans = con(matrix)
    return con(matrix_trans).astype(np.complex64)


def band_xz(kx,ky):
    eigvalue,eigvector = eig(hamiltonian_matrix_xz(kx,ky))
    idx = (eigvalue.real).argsort()[::-1]
    eigvalue = eigvalue[idx]
    eigvector = eigvector[:,idx]
    return eigvalue,eigvector
band_vec_xz = vec(band_xz, signature='(),()->(m),(m,m)')

def band_yz(kx,ky):
    eigvalue,eigvector = eig(hamiltonian_matrix_yz(kx,ky))
    idx = (eigvalue.real).argsort()[::-1]
    eigvalue = eigvalue[idx]
    eigvector = eigvector[:,idx]
    return eigvalue,eigvector
band_vec_yz = vec(band_yz, signature='(),()->(m),(m,m)')

def math_func_value(arr0,arr1):
    arr0_row = arr0.reshape(num_sublattice,1)
    arr1_row = arr1.reshape(num_sublattice,1)
    arr=np.concatenate((arr0_row,arr1_row),axis=1)
    arr.shape = num_sublattice*orbit_number
    return arr
from scipy.sparse import bsr_matrix
def math_func_vector(arr0,arr1):
    arr = np.array([[arr0,np.zeros((num_sublattice,num_sublattice))],[np.zeros((num_sublattice,num_sublattice)),arr1]])
    arr = arr.transpose((0,2,1,3)).reshape(6,6)
    arr3 = arr[:,[0,3,1,4,2,5]][[0,3,1,4,2,5]]
    return arr3

def band_vector(kx,ky):
    return math_func_vector(band_xz(kx,ky)[1],band_yz(kx,ky)[1])
band_vec_plot = vec(band_vector, signature='(),()->(m,m)')


