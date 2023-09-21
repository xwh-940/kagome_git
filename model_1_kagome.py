from parameters import *
from filename import *
from math_func import *

t_xz = 0.5*2
mu = para.mu
e_xz = -0.055*2 + mu 

orbit_number = 1
sublattice_number = 3
band_number = orbit_number * sublattice_number
band_number_square = band_number ** 2

print_and_save_txt("---------parameters---------------")
print_and_save_txt("number of k, wn:", kn, Tn)
print_and_save_txt("temperature:", T)
print_and_save_txt("t, mu:", t_xz, e_xz)
print_and_save_txt("Hubbard term U:", U0)
print_and_save_txt("---------parameters---------------")
print_and_save_txt("                                  ")


kx = ky = k_orgin_1

#ky_basis = np.array([0, 1])
#kx_basis = np.array([sqrt(3)/2, -1/2]) # true
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
    # return 1.0 + np.exp(-num*k_vector@arr_a)
    return 2*np.cos(np.complex64(1.0)*k_vector@arr_a)

"""
@njit
def Phi(i, j, kx,ky):
    '''
    lattice structure factors
    A, B, C: i = 0, 1, 2
    '''
    if i==j:
        return 0
    if i==0 and j==1:
        #a1
        return factor(kx, ky, 1/2, np.sqrt(3)/2)
    if i==1 and j==2:
        #a3
        return factor(kx, ky, -1, 0)
    if i==0 and j==2:
        #a2
        return factor(kx, ky, -1/2, np.sqrt(3)/2)
    else:
        return np.conj(Phi(j, i, kx, ky))
"""
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
def hamiltonian(i,j,kx,ky):
    return e_xz*dirac_func(i,j) - t_xz*Phi(j,i,kx,ky)
#hamiltonian_vec = vec(hamiltonian, signature='(),(),(),(),(),()->()', otypes=[complex])
sublattice_arr = con(np.arange(sublattice_number))

num_sublattice = sublattice_arr.shape[0]
num_band = num_sublattice
sublattice_arr1 = sublattice_arr.reshape(num_sublattice,1)
sublattice_arr2 = sublattice_arr.reshape(1,num_sublattice)

@njit
def hamiltonian_matrix(kx,ky):
    matrix=hamiltonian(sublattice_arr1, sublattice_arr2, kx, ky)
    matrix_trans = con(matrix)
    return con(matrix_trans).astype(np.complex64)

def band(kx,ky):
    eigvalue,eigvector = eig(hamiltonian_matrix(kx,ky))
    idx = (eigvalue.real).argsort()[::-1]
    eigvalue = eigvalue[idx]
    eigvector = eigvector[:,idx]
    return eigvalue,eigvector
band_vec = vec(band, signature='(),()->(m),(m,m)')

def band_value(kx,ky):
    eigvalue,eigvector = eig(hamiltonian_matrix(kx,ky))
    idx = (eigvalue.real).argsort()[::-1]
    eigvalue = eigvalue[idx]
    return eigvalue.real
band_value_vec = vec(band_value, signature='(),()->(m)')

'''
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
ax = plt.axes(projection='3d')
#def band_plot(plot_data):
ax.plot_surface(kx_band, ky_band, band_vec(kx_band, ky_band)[0][:,:,0])
plt.show()
'''


@jit(nopython=True)
def green0(wn0, kx0, ky0,fre):
    "2*2 matrix of bare green's function with given wn,kx,ky,kz"
    #-dirac_e*(fre-1)*np.identity(band_number)*1j
    green00 = inv(fre*wn0*1j*np.identity(band_number)-hamiltonian_matrix(kx0, ky0)-dirac_e*(fre-1)*np.identity(band_number)*1j)
    #green00 = inv(wn0*1j*np.identity(band_number)-hamiltonian_matrix(kx0, ky0))
    return green00

@jit(nopython=True, nogil=True)
def green0m(fre):
    "array(Tn,kn,kn,kn) of bare green's function with given orbit "
    green0m0 = np.empty((Tn, kn, kn, band_number, band_number), dtype=np.complex64)
    for para in prange (para_array.shape[0]): 
        green0m0[para_array[para, 0],para_array[para, 1],para_array[para, 2]] = \
        green0(wn[para_array[para, 0]], kx[para_array[para, 1]], ky[para_array[para, 2]],fre)
    return  green0m0
 

def green0me(a1,a2):
    "no compution:array(Tn,kn,kn,kn) of bare green's function with given orbit "
    return np.load(green0m_str(s(a1),s(a2)))


@jit(nopython=True)
def green0_inverse(wn0,kx0,ky0):
    green_inverse0 =wn0*1j*np.identity(band_number)-hamiltonian_matrix(kx0,ky0)
    return green_inverse0

@jit(nopython=True,nogil=True)
def green0_inversem():
    "bare green0_inverse with given orbit"
    green_inversem0 = np.empty((Tn, kn, kn, band_number, band_number),dtype=np.complex64)
    for para in prange (para_array.shape[0]):
        green_inversem0[para_array[para,0],para_array[para,1],para_array[para,2]] = \
        green0_inverse(wn[para_array[para,0]],kx[para_array[para,1]],ky[para_array[para,2]])
    return  green_inversem0               

#in order to have less compution

  

def green0_inverseme(a1,a2):
    "the last green0_inverse with no compution"
    return np.load(green0_inversem_str(s(a1),s(a2)))


def transf42(func,b1,b2,*arg):
    "6*6*6*6 -> 36*36 "
    num = band_number
    return func(b1//num,b1%num,b2//num,b2%num,*arg)

def transf24(func,a1,a2,a3,a4,*arg):
    "36*36 -> 6*6*6*6 "
    num = band_number
    return func(num*a1+a2,num*a3+a4,*arg)

num_orbit=1
def Us_func(ii,jj,kk,ll):
    ii_sub = ii//num_orbit
    jj_sub = jj//num_orbit
    kk_sub = kk//num_orbit
    ll_sub = ll//num_orbit
    if ii_sub==jj_sub and jj_sub==kk_sub and kk_sub==ll_sub:
        return U0
    return 0

def Uc_func(ii,jj,kk,ll):
    ii_sub = ii//num_orbit
    jj_sub = jj//num_orbit
    kk_sub = kk//num_orbit
    ll_sub = ll//num_orbit
    if ii_sub==jj_sub and jj_sub==kk_sub and kk_sub==ll_sub:
        return U0
    return 0

def Us_matrix(b1,b2):
    return transf42(Us_func,b1,b2)

def Uc_matrix(b1,b2):
    return transf42(Uc_func,b1,b2)

Us = np.array([[Us_matrix(i,j) for j in \
   range(band_number_square)] for i in \
   range(band_number_square)]).astype(np.complex64)
Uc = np.array([[Uc_matrix(i,j) for j in \
   range(band_number_square)] for i in \
   range(band_number_square)]).astype(np.complex64)

def density(green_data, n):
    return f(np.real( 2*((T/(kn*kn))*np.sum(green_data(n,n))+0.5)))

# band-plot
road1 = [np.array([ky/sqrt(3),ky])*pi for ky in np.arange(0,2/sqrt(3),0.01)]
road2 = [np.array([kx,2/sqrt(3)])*pi for kx in np.arange(2/3,0,-0.01)]
road3 = [np.array([0,ky])*pi for ky in np.arange(2/sqrt(3),0,-0.01)]
road = np.array(road1 + road2 + road3)/2

def plot_band():
    road_60 = tran_96_vec(road)
    distance_road = distance(road)

    band_road = band_vec(road_60[:,0],road_60[:,1])
    fig, ax = plt.subplots()
    ax.set_title('band')
    ax.set_ylabel('E/eV')
    ax.set_xlim(distance_road[0],distance_road[-1])
    #ax.set_ylim(-3.5,3.5)
    ax.set_xticks([distance_road[0], distance_road[116], distance_road[183], distance_road[-1]])
    ax.set_xticklabels([r"$\Gamma$", r"$K$", r"$M$", r"$\Gamma$"])
    ax.axhline(y=0, c="blue", linestyle='--',linewidth=2)
    ax.axvline(x=distance_road[116], c="red", linestyle="--", linewidth=2)
    ax.axvline(x=distance_road[183], c="red", linestyle="--", linewidth=2)
    for i in range(num_band):
        ax.scatter(distance_road,band_road[0][:,i].real, c='black', s=10)
    fig.savefig(path_model_fig+'//band.jpg',dpi=300)
    plt.close()


