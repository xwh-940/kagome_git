from model_1_kagome import *
from fs_point_data import *

def band_vector(k_60):
    return band(k_60[0]*np.pi,k_60[1]*np.pi)
band_vector_vec = vec(band_vector,signature='(n)->(m),(m,m)')

# Fermi velocity
def given_band_vals(k_90):
    # k -> 90
    return band_vals_fs(k_90)[which_band]

def gradient_mod(x0):
    grad = approx_fprime(x0, given_band_vals, epsilon=1e-4)  # 计算梯度
    mod = np.linalg.norm(grad, ord=2)
    return mod
gradient_mod_vec = np.vectorize(gradient_mod,signature="(n)->()")

# print(1/gradient_mod_vec(fs_points_90))

kx_60 = k_orgin[1:].copy()  # x坐标 k_60 -2/sqrt(3)~2/sqrt(3) no pi
ky_60 = k_orgin[1:].copy()  # y坐标 k_60 
def gamma_f_orbit(a1,a2,a3,a4,V_arr):
    'V_arr is flipped to -2/sqrt(3)~2/sqrt(3)'
    gamma0 = RectBivariateSpline(kx_60, ky_60, V_arr[band_number*a1+a2,band_number*a3+a4].real) 
    return gamma0

def gamma_f_q(a1,a2,a3,a4,q,gamma_f_orbit_arr):
    return np.real(gamma_f_orbit_arr[a1,a2,a3,a4](q[0],q[1])[0,0])
gamma_f_q_vec = vec(gamma_f_q,signature='(),(),(),(),(n),(m,m,m,m)->()')

def tranform_a(a1,a4,k_60):
    return band_vector(k_60)[1][a1,which_band] * band_vector(-k_60)[1][a4,which_band]
tranform_a_vec = vec(tranform_a,signature='(),(),(n)->()')

a1_arr = np.arange(band_number).reshape((band_number,1,1,1))
a2_arr = np.arange(band_number).reshape((1,band_number,1,1))
a3_arr = np.arange(band_number).reshape((1,1,band_number,1))
a4_arr = np.arange(band_number).reshape((1,1,1,band_number))
# identity_3 = np.zeros((band_number,band_number,band_number,band_number))
# for a1 in range(band_number):
#     for a2 in range(band_number):
#         for a3 in range(band_number):
#             for a4 in range(band_number):
#                 identity_3[a1,a2,a3,a4] = 1

def gamma_f(k1_60,k2_60,gamma_f_orbit_arr):
    q1_60 = period_vec(k1_60-k2_60,-1*multi*(1/sqrt(3)),1*multi*(1/sqrt(3)))
    # q1_60 = period_vec(q1_60,-1*(1/sqrt(3)),1*(1/sqrt(3)))
    # _gamma = gamma_f_q_vec(a1_arr,a2_arr,a3_arr,a4_arr,q1_60,gamma_f_orbit_arr) * identity_3
    _gamma = gamma_f_q_vec(a1_arr,a2_arr,a3_arr,a4_arr,q1_60,gamma_f_orbit_arr)
    gamma_ij = (np.real(_gamma) * tranform_a_vec(a2_arr,a3_arr,k1_60) * tranform_a_vec(a1_arr,a4_arr,k2_60).conj()).sum()
    # gamma_ij = np.real(_gamma).sum()
    return gamma_ij
gamma_f_vec = vec(gamma_f,signature='(n),(n),(m,m,m,m)->()')


def plot_fs_point_phi(phi_arr,path,name,number_s,vec_num):
    ax = plt.subplot(111)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'$k_{x}/\pi$')
    ax.set_ylabel(r'$k_{y}/\pi$')
    ax.set_xlim(-0.75*multi,0.75*multi)
    ax.set_ylim(-0.75*multi,0.75*multi)
    second_bz = plt.Polygon(kagome_lattice, fill=None, edgecolor='black')
    ax.add_patch(second_bz)
    first_bz = plt.Polygon(kagome_lattice_bz, fill=None, edgecolor='black')
    ax.add_patch(first_bz)
    band_value_data = band_value_vec(kx_band_fs,ky_band_fs)
    #ax.scatter(fs_points_90[:,0],fs_points_90[:,1],s=number_s)
    vmax_scatter,vmin_scatter = np.abs(phi_arr[:,vec_num]).max(), -np.abs(phi_arr[:,vec_num]).max()
    ff=ax.scatter(fs_points_90[:,0],fs_points_90[:,1],s=number_s,c=phi_arr[:,vec_num].real,cmap='seismic',vmax=vmax_scatter,vmin=vmin_scatter)
    plt.colorbar(ff)
    fig1 = plt.gcf() #gcf: Get Current Figure
    fig1.savefig(path+"//"+name+".png", dpi=800)
    plt.close()

# def gamma_f_back(k1,k2):
#     q1 = period_vec(k1-k2,-1*multi*(1/sqrt(3)),1*multi*(1/sqrt(3)))
#     q2 = period_vec(k1+k2,-1*multi*(1/sqrt(3)),1*multi*(1/sqrt(3)))
#     gamma_ij = 0
#     for a1 in range(band_number):
#         for a2 in range(band_number):
#             for a3 in range(band_number):
#                 for a4 in range(band_number):
#                     _gamma1=np.real(gamma_f_orbit(a1,a2,a3,a4)(q1[0],q1[1])[0,0])
#                     _gamma2=np.real(gamma_f_orbit(a1,a2,a3,a4)(q2[0],q2[1])[0,0])
#                     _gamma=(_gamma1+_gamma2)/2
#                     _transform0 = band_vector(k1)[1][a2,which_band].conj() * band_vector(-k1)[1][a3,which_band].conj()
#                     _transform1 = band_vector(k2)[1][a1,which_band] * band_vector(-k2)[1][a4,which_band]
#                     gamma_ij += np.real(_gamma) * _transform0 * _transform1
#     return gamma_ij
