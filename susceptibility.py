from plot_point import *
from model_1_kagome import *
id4 = np.identity(band_number_square).astype(np.complex64)

def chi0f(a1,a2,a3,a4,greenf):
    "bare chi0 with no renormalization"
    # g0_r1 = fft.fftn(greenf(a3,a1))
    # g0_ir2 = fft.ifftn(greenf(a2,a4))
    g0_r1 = fft.ifftn(greenf(a3,a1))
    g0_ir2 = fft.fftn(greenf(a2,a4))   
    chi00_r =ne("-(g0_r1*g0_ir2)*Tn*T").astype(np.complex64)
    chi0f0  = fft.ifftn(chi00_r)
    chi0f0 = chi0f0
    # return np.abs(chi0f0[0]).copy().astype(np.complex64)
    return chi0f0[0].copy()

def chi0m_save(greenf):
    chi0m = np.array([[transf42(chi0f,i,j,greenf) for j in range(band_number**2)] for i in range(band_number**2)])
    np.save(data+"//"+chi0_matrix_str,chi0m)
    return 0

@njit
def chis(chif_matrix,kx0,ky0):
    "spin of susceptibility"
    chim = con(chif_matrix[:,:,kx0,ky0])
    chis0 = con(inv(id4 - chim @ Us)) @ chim
    return chis0

@njit
def chic(chif_matrix,kx0,ky0):
    "charge of susceptibility"
    chim = con(chif_matrix[:,:,kx0,ky0])
    chic0 = con(inv(id4 + chim @ Uc)) @ chim
    return chic0   

@njit
def chis_arr(chif_matrix):
    "vertex IP2 funtion of single channel with given orbit" 
    chis10 = np.empty((kn,kn,band_number**2,band_number**2),dtype=np.float32)
    for para in prange(para_array_kn.shape[0]):
        chis10[para_array_kn[para,0],para_array_kn[para,1]]\
            = np.real(chis(chif_matrix,para_array_kn[para,0],para_array_kn[para,1]))
    return  chis10 

@njit
def chic_arr(chif_matrix):
    "vertex IP2 funtion of single channel with given orbit" 
    chis10 = np.empty((kn,kn,band_number**2,band_number**2),dtype=np.float32)
    for para in prange(para_array_kn.shape[0]):
        chis10[para_array_kn[para,0],para_array_kn[para,1]]\
            = np.real(chic(chif_matrix,para_array_kn[para,0],para_array_kn[para,1]))
    return  chis10 


def chif_plot(chi_data, path, name):
    chi0_arr_sum = np.zeros((kn,kn),dtype=np.complex64)
    num_s = 130*(32/kn)*0.5
    for i  in range(int(band_number**2)):
        a1_ = i//band_number
        b1_ = i%band_number
        a1_r = band_number*a1_+a1_
        b1_r = band_number*b1_+b1_
        ax = plt.subplot(111)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(r'$k_{x}/\pi$')
        ax.set_ylabel(r'$k_{y}/\pi$')
        ax.set_xlim(-0.75*multi,0.75*multi)
        ax.set_ylim(-0.75*multi,0.75*multi)
        chi0_arr_sum += chi_data[a1_r,b1_r]
        ff=plt.scatter(kx_point,ky_point, c=np.real(chi_data[a1_r,b1_r]), alpha=0.5, cmap='rainbow',s=num_s,marker='.')
        plt.colorbar(ff)
        second_bz = plt.Polygon(kagome_lattice, fill=None, edgecolor='black')
        ax.add_patch(second_bz)
        first_bz = plt.Polygon(kagome_lattice_bz, fill=None, edgecolor='black')
        ax.add_patch(first_bz)

        fig1 = plt.gcf()  #gcf: Get Current Figure
        fig1.savefig(path+"//"+name+"_"+s(i//band_number)+s(i%band_number)+".png", dpi=300)
        plt.close()
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
    ff=ax.scatter(kx_point,ky_point, c=np.real(chi0_arr_sum), alpha=0.5,cmap='rainbow',s=num_s,marker='.')
    plt.colorbar(ff)
    fig2 = plt.gcf()  #gcf: Get Current Figure
    fig2.savefig(path+"//"+name+"_sum.png", dpi=300)
    plt.close()

def chif_ijij_plot(chi_data, path, name):
    chi0_arr_sum = np.zeros((kn,kn),dtype=np.complex64)
    num_s = 130*(32/kn)*0.5
    for i  in range(int(band_number**2)):
        a1_ = i//band_number
        b1_ = i%band_number
        a1_r = band_number*a1_+a1_
        b1_r = band_number*b1_+b1_
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
        chi0_arr_sum += chi_data[i,i]
        ff=ax.scatter(kx_point,ky_point, c=np.real(chi_data[i,i]), alpha=0.5, cmap='rainbow',s=num_s,marker='.')
        plt.colorbar(ff)
        fig1 = plt.gcf()  #gcf: Get Current Figure
        fig1.savefig(path+"//"+name+"_"+s(i//band_number)+s(i%band_number)+"_ijij.png", dpi=300)
        plt.close()
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
    ff=ax.scatter(kx_point,ky_point, c=np.real(chi0_arr_sum), alpha=0.5,cmap='rainbow',s=num_s,marker='.')
    plt.colorbar(ff)
    fig2 = plt.gcf()  #gcf: Get Current Figure
    fig2.savefig(path+"//"+name+"_sum_ijij.png", dpi=300)
    plt.close()
