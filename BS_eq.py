from plot_point import *
from plot_point_fs import *
from math_func import *
from susceptibility import *
#from plot_band_orbit_2 import *

def G_q(greenf,a1,a2,a3,a4):
    "g(q)*g(-q)"
    #G0_q = greenf(a1,a2)*tr.flip(greenf(a3,a4),[0,1]) 
    G0_q = greenf(a1,a2)*tr.conj(greenf(a3,a4))
    return G0_q
 
def hh(greenf,a1,b1,a4,b2,phi_f):
    "g(q)*g(-q)* phi(q)"
    hh0 = G_q(greenf,a1,b1,a4,b2) * phi_f[b1,b2] 
    return hh0

def phi_value(greenf,V_bs_func,phi_f):
    phi_f_0 = tr.tensor(phi_f).cuda()
    phi_f_0 = phi_f_0.reshape((band_number,band_number,kn,kn))
    def phi(a2,a3):
        phi21 = tr.zeros((kn,kn),dtype=tr.float32).cuda()
        for b1 in range(band_number):
            for b2 in range(band_number):
                for a1 in range(band_number):
                    for a4 in range(band_number):                        
                        Vs1_r = V_bs_func(a1,a2,a3,a4)
                        #hh_r  = hhme(a1,b1,a4,b2,n)
                        hh_r  = tr.fft.fftn(hh(greenf,a1,b1,a4,b2,phi_f_0))
                        phi20_r = - (Vs1_r * hh_r)*(T/(kn*kn))
                        phi20 = tr.fft.ifftn(phi20_r)
                        phi21 +=  phi20.real
        phi21 = symmetry_chance(phi21,a2,a3)
        #print("phi21:",phi21[20,20])
        #phi21 += lamda * phi_f[a2,a3]
        #phi21 = phi21/(1+lamda)
        return phi21
    
    phi_output = tr.zeros((band_number,band_number,kn,kn),dtype=tr.float32).cuda()
    for orbit in prange(orbitral_array.shape[0]):
        phi_output[orbitral_array[orbit,0],orbitral_array[orbit,1]] \
            = phi(orbitral_array[orbit,0],orbitral_array[orbit,1])
        #np.save(phi_str(s(orbitral_array[orbit,0]),s(orbitral_array[orbit,1])), \
        #        phi_output[orbitral_array[orbit,0],orbitral_array[orbit,1]])
    print("hello")
    return np.real((phi_output.reshape(band_number*band_number*kn*kn).cpu().numpy()))


'''
@vectorize
def phik0(kx,ky):
    return np.cos(2.3*kx+1) - np.cos(ky+0.5) + kx

def phik11(a1,a2):
    kk = k_orgin_1

    phik11_0 = phik0(kk[para_array_kn[:,0]],kk[para_array_kn[:,1]])
    phik11_0.shape = kn,kn
    return phik11_0.astype(np.complex64)
'''
# zero-frequency

# mod of normalization
# a = 0 or 1
def mod_func(a):
    phi_ordi = tr.load(phi_str(s(a),s(a)))
    return tr.sqrt((phi_ordi.conj() * phi_ordi).real.sum()).item()

def plot_fs(dirac_k,path,name,number_s):
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
    for band_num in range(band_number):
        arg_band = np.where(np.abs(band_value_data[:,:,band_num]) < dirac_k)
        ax.scatter(kx_point_fs[arg_band],ky_point_fs[arg_band],s=number_s)
    fig1 = plt.gcf() #gcf: Get Current Figure
    fig1.savefig(path+"//"+name+".png", dpi=800)
    plt.close()


def plot_phi(phi_data, path, name):
    band_value_data = band_value_vec(kx_band_fs,ky_band_fs)
    num_s = 130*(0.5/kn)*8
    for i  in range(band_number):
        #for j in range(band_number):
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
        f=ax.scatter(kx_point_phi,ky_point_phi,c=np.real(phi_data[i,i,:,:]), cmap='rainbow',s=num_s,alpha=1)
        plt.colorbar(f)
        for band_num in range(band_number):
            arg_band = np.where(np.abs(band_value_data[:,:,band_num]) < 0.05)
            ax.scatter(kx_point_fs[arg_band],ky_point_fs[arg_band],s=1,alpha=0.5)
        fig1 = plt.gcf() #gcf: Get Current Figure
        fig1.savefig(path+"//"+name+"_"+s(i)+s(i)+".png", dpi=800)
        plt.close()

def plot_phi_band(phi_data, path, func, name):
    band_value_data = band_value_vec(kx_band_fs,ky_band_fs)
    "phi_data:o,o,kn,kn"
    num_s = 130*(0.5/kn)*8
    #transf_mat = band_vec_plot(kx_band, ky_band)
    transf_mat = band_vec(kx_band, ky_band)[1]
    #transf_mat_mu = band_vec(-kx_band, -ky_band)[1]
    #transf_mat_conj = np.conj(transf_mat.transpose(0,1,3,2))
    phi_data_0 = phi_data.transpose(2,3,0,1)
    phi_plot_data = np.conj(transf_mat.transpose(0,1,3,2))@phi_data_0@transf_mat
    #phi_plot_data = transf_mat@phi_data_0@transf_mat_mu
    for i  in range(band_number):
        #for j in range(band_number):
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
        vmax_scatter,vmin_scatter = np.abs(func(phi_plot_data[:,:,i,i])).max(), -np.abs(func(phi_plot_data[:,:,i,i])).max()
        f=ax.scatter(kx_point_phi,ky_point_phi,c=func(phi_plot_data[:,:,i,i]), cmap='rainbow',s=num_s, alpha=1,vmax=vmax_scatter,vmin=vmin_scatter)
        plt.colorbar(f)
        for band_num in range(band_number):
            arg_band = np.where(np.abs(band_value_data[:,:,band_num]) < 0.05)
            ax.scatter(kx_point_fs[arg_band],ky_point_fs[arg_band],s=1,alpha=0.5)
        fig1 = plt.gcf() #gcf: Get Current Figure
        fig1.savefig(path+"//"+name+"_"+s(i)+s(i)+".png", dpi=800)
        plt.close()

'''
def fix_transf(transf_mat):
    'input:kn*kn*band_num*band_num'
    transf_mat0 = transf_mat.reshape((kn*kn,band_number,band_number))
    zero_matrix = np.zeros((1,band_number,band_number))
    transf_mat_1 =  np.concatenate((transf_mat0,zero_matrix),axis=0)
    transf_mat_2 =  np.concatenate((zero_matrix,transf_mat0),axis=0)
    transf_data = (transf_mat_1.transpose(0,2,1)@transf_mat_2)[1:-1].diagonal(axis1=1,axis2=2)
    print(np.sort(np.abs(transf_data[:,0])))
    transf_data_1 = np.concatenate((np.ones((1,band_number)),transf_data),axis=0)
'''    
