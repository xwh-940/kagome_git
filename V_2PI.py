from susceptibility import *

@njit
def V_singlet(chif_matrix,kx0,ky0):
    "vertex IP2 funtion of single channel"
    Vs0 = v15 * Us @ chis(chif_matrix,kx0,ky0) @ Us\
        - v05 * Uc @ chic(chif_matrix,kx0,ky0) @ Uc\
        + v05 * (Uc + Us)
    return Vs0

@njit
def V_triplet(chif_matrix,kx0,ky0):
    "vertex IP2 funtion of single channel"
    Vs0 = - v05 * Us @ chis(chif_matrix,kx0,ky0) @ Us\
          - v05 * Uc @ chic(chif_matrix,kx0,ky0) @ Uc\
          + v05 * (Uc + Us)
    return Vs0


if para.channel == "singlet":
    V_channel = V_singlet
if para.channel == "triplet":
    V_channel = V_triplet


@njit
def V_bs(chif_matrix):
    "vertex IP2 funtion of single channel with given orbit" 
    Vs10 = np.empty((kn,kn,band_number_square,band_number_square),dtype=np.complex64)
    for para in prange(para_array_kn.shape[0]):
        Vs10[para_array_kn[para,0],para_array_kn[para,1]]\
            = np.real(V_channel(chif_matrix,para_array_kn[para,0],para_array_kn[para,1]))
    return  Vs10 

