from green_re import *
from V_2PI import *
from BS_eq import *
# from fs_point_data import *
# from Eliashberg import *

# plot_band()
# plot_fs(dirac_k=0.02,path=path_model_fig,name="fs",number_s=3)
# plot_fs_point(path=path_model_fig,name="fs_new",number_s=3)

# 用解析结果计算磁化率
# input：60度坐标系下的bz，不含pi

#@njit
def fermi_distribution(ek_arr):
    _func = 1/(1+tr.exp(ek_arr/T))
    return _func

def trans_q(arr,qxn,qyn):
    'e(k)(128*128) -> e(k+q)(128*128) qxn:0~127'
    return tr.roll(arr,shifts=(qxn,qyn),dims=(-2,-1))
trans_q_vec = vec(trans_q,signature='(m,m),(),()->(m,m)')

k_ek, k_a = band_vec(kx_band,ky_band) # 128 128 3, 128 128 3 3
k_ek =  tr.tensor(k_ek.transpose((2,0,1)),dtype=tr.complex128).cuda()
k_a =  tr.tensor(k_a.transpose((2,3,0,1)),dtype=tr.complex128).cuda()
e_mu_k = k_ek.reshape(1,1,1,1, band_number,1, kn,kn)
n_e_mu_k = fermi_distribution(e_mu_k)

a_mu_4_k = k_a.reshape(1,1,1,band_number, band_number,1, kn,kn)
a_mu_2_k = k_a.reshape(1,band_number,1,1, band_number,1, kn,kn)


eta = 0.0001



# _element1 = tr.ones((band_number,band_number,band_number,band_number,band_number,band_number,kn,kn)).cuda()
def bare_sus(qxn,qyn):
    kq_ek = trans_q(k_ek,qxn,qyn)
    kq_a = trans_q(k_a,qxn,qyn)
    # kq_ek, kq_a = band_vec(kx_band_q,ky_band_q)

    e_nv_kq = kq_ek.reshape(1,1,1,1, 1,band_number, kn,kn) + eta*1j
    n_e_nu_kq = fermi_distribution(e_nv_kq)
    # n_e_mu_kq = trans_q(n_e_mu_k,qxn,qyn)

    a_nv_1_kq = kq_a.reshape((band_number,1,1,1, 1,band_number, kn,kn))
    a_nv_3_kq = kq_a.reshape((1,1,band_number,1, 1,band_number, kn,kn))
    
    _element1 = a_mu_4_k * a_mu_2_k.conj() * a_nv_1_kq * a_nv_3_kq.conj()
    
    _element2 = (n_e_mu_k - n_e_nu_kq)/(e_mu_k - e_nv_kq)
    _result = -(_element1 * _element2).sum(dim=(4,5,6,7)) / (kn*kn)
    # return np.complex64(np.abs(_result.cpu().numpy()))
    return np.complex64(_result.cpu().numpy())
bare_sus_vec = vec(bare_sus,signature='(),()->(m,m,m,m)')
kn_arr = np.arange(kn)

time1 = time.time()
# print("test:",bare_sus(21,10)[0,0,0,0]) 
# # bare_sus(21,10)
# sys.exit()
arr_result = bare_sus_vec(kn_arr.reshape((kn,1)),kn_arr.reshape((1,kn))).transpose((2,3,4,5,0,1)).reshape((band_number**2,band_number**2,kn,kn))
print(arr_result.shape)
np.save("arr_result.npy",arr_result)
time2 = time.time()
print("time:",time2 - time1)

chis_test = chis_arr(arr_result).transpose(2,3,0,1)
np.save("chis.npy",chis_test)
chif_plot(chis_test,path_chi0s_fig,"chi0s")
# chif_ijij_plot(chis_test,path_chi0s_fig,"chi0s")

    
