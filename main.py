from green_re import *
from V_2PI import *
from BS_eq import *
from fs_point_data import *
from Eliashberg import *

plot_band()
plot_fs(dirac_k=0.02,path=path_model_fig,name="fs",number_s=3)
plot_fs_point(path=path_model_fig,name="fs_new",number_s=3)

# from itertools import permutations
# l_arr = [a1_arr,a2_arr,a3_arr,a4_arr]
# l_number = [0,1,2,3]
# for iii in list(permutations(l_number)):
#     test_arr = tranform_a_vec(l_arr[iii[0]],l_arr[iii[1]],fs_points[0]).conj() * tranform_a_vec(l_arr[iii[2]],l_arr[iii[3]],fs_points[86])
#     np.save(str(iii)+"test_arr.npy",test_arr)

# sys.exit()

# fff=plt.plot(gradient_mod_vec(fs_points_90))

# fig1 = plt.gcf() #gcf: Get Current Figure
# fig1.savefig("gradient_mod_vec(fs_points_90).png", dpi=800)
# plt.close()

# sys.exit()

start0 = time.time()
green0m_copy = con(green0m(1.0).transpose((3,4,0,1,2)))
print("green0m_copy dtype",green0m_copy.dtype)
for i in range(band_number):
    for j in range(band_number):
        np.save(green0m_str(s(i),s(j)),green0m_copy[i,j,:,:,:])
del green0m_copy
elapsed0 = f(time.time() - start0)
print_and_save_txt("green0 Time used:",elapsed0)   
print_and_save_txt("                                  ")

density_sum = np.array([density(green0me, n) for n in range(band_number)]).sum()
print_and_save_txt("density :", density_sum)
print_and_save_txt("                                  ")

chi0m_save(green0me)
chi0_matrix = np.load(data+"//"+"chi0m.npy")
chif_plot(chi0_matrix,path_chi0_fig,"chi0")  # RPA时chi0_matrix[:,:,0,:,:]变为chi0_matrix
chif_ijij_plot(chi0_matrix,path_chi0_fig,"chi0")  # RPA时chi0_matrix[:,:,0,:,:]变为chi0_matrix
np.save(SOURCE+"//"+name_time+"//"+chis_str, chis_arr(chi0_matrix))
chif_plot(chis_arr(chi0_matrix).transpose(2,3,0,1),path_chi0s_fig,"chi0s")
chif_ijij_plot(chis_arr(chi0_matrix).transpose(2,3,0,1),path_chi0s_fig,"chi0s")
# sys.exit()
#V_2PI计算---begin---
start2 = time.time()
chi0r_matrix = chi0_matrix



test_array = flip_f_vec(chis_arr(chi0_matrix).transpose(2,3,0,1))
# test_array = flip_f_vec(chi0_matrix)
gamma_f_orbit_arr = np.zeros((band_number,band_number,band_number,band_number),dtype=object)
for a1 in range(band_number):
        for a2 in range(band_number):
            for a3 in range(band_number):
                for a4 in range(band_number):
                    gamma_f_orbit_arr[a1,a2,a3,a4] = gamma_f_orbit(a1,a2,a3,a4,test_array)

def test_gamma(q1_60_x, q2_60_y):
    q1_60 = np.array([q1_60_x, q2_60_y])
    arr = gamma_f_q_vec(a1_arr,a2_arr,a3_arr,a4_arr,q1_60,gamma_f_orbit_arr)
    arr_33 = np.zeros((band_number,band_number))
    for ib in range(band_number):
        for jb in range(band_number):
            arr_33[ib,jb] = arr[ib,ib,jb,jb]
    # print(arr)
    eigvalue33,eigvector33 = eig(arr_33)
    # return eigvalue33,eigvector33
    return eigvalue33
# print(test_gamma(0,0)[0])
# print(test_gamma(0,0)[1])
# sys.exit()
test_gamma_vec = vec(test_gamma,signature='(),()->(m)')
test_gamma_arr=test_gamma_vec(kx_band/pi, ky_band/pi).real

def plot_vs_band():
    road_60 = tran_96_vec(road)/pi
    distance_road = distance(road)

    band_road = test_gamma_vec(road_60[:,0],road_60[:,1])
    #print(band_road.real)
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
        ax.scatter(distance_road,band_road[:,i].real, c='black', s=10)
    fig.savefig(path_model_fig+'//vs_band.jpg',dpi=300)
    plt.close()
plot_vs_band()

def plot_test_gamma(gamma_data, path, name):
    num_s = 130*(32/kn)
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
    ff=ax.scatter(kx_point_phi,ky_point_phi, c=gamma_data, alpha=0.5, cmap='rainbow',s=num_s,marker='.')
    plt.colorbar(ff)
    fig1 = plt.gcf()  #gcf: Get Current Figure
    fig1.savefig(path+"//"+name+".png", dpi=300)

for a1_b in range(band_number):
    plot_test_gamma(test_gamma_arr[:,:,a1_b],path_phi_fig,"chis_eig_"+str(a1_b))


# sys.exit()


green_fre = green0m(0.0)
max_gf=green_fre[int(Tn/2)].max()
# print_and_save_txt("gf_max",max_gf)
# print_and_save_txt("gf_max_**2",max_gf**2)
green_in = tr.zeros((band_number,band_number,kn,kn),dtype=tr.complex64).cuda()
for orbit in prange ( orbitral_array.shape[0] ):
    green_in[orbitral_array[orbit,0],orbitral_array[orbit,1]]\
        = tr.tensor(green_fre[int(Tn/2),:,:,(orbitral_array[orbit,0]),(orbitral_array[orbit,1])]).cuda()
def greenrme(a1,a2):
    return green_in[a1,a2]

Vs1_copy = tr.tensor(V_bs(chi0r_matrix)).cuda()
# for i1 in range(band_number):
#     for j1 in range(band_number):
#         for k1 in range(band_number):
#             for l1 in range(band_number):
#                 tr.save(tr.fft.fftn(Vs1_copy[:,:,band_number*i1+j1,band_number*k1+l1]),V_bs_fft_str(s(i1),s(j1),s(k1),s(l1)))
V_sum = Vs1_copy.cpu().numpy().transpose((2,3,0,1))
print_and_save_txt("V_sum.shape:", V_sum.shape)
del Vs1_copy

def V_bs_me(a1,a2,a3,a4):
    return tr.load(V_bs_fft_str(s(a1),s(a2),s(a3),s(a4)))
elapsed2 = f(time.time() - start2)
print_and_save_txt("Vs Time used:",elapsed2)
print_and_save_txt("                                  ")






# fermi surface
start3 = time.time()
# V_sum_fft = np.fft.fftn(V_sum,axes=(2,3))
chif_plot(V_sum, path_chi0s_fig,"V_s")
chif_ijij_plot(V_sum, path_chi0s_fig,"V_s")
# sys.exit()
fs_points_number = fs_points.shape[0]

fermi_velocity = 1/gradient_mod_vec(fs_points_90) # shape=(256,)
flip_V_sum = flip_f_vec(V_sum)


gamma_f_orbit_arr = np.zeros((band_number,band_number,band_number,band_number),dtype=object)
for a1 in range(band_number):
        for a2 in range(band_number):
            for a3 in range(band_number):
                for a4 in range(band_number):
                    gamma_f_orbit_arr[a1,a2,a3,a4] = gamma_f_orbit(a1,a2,a3,a4,flip_V_sum)
# print(gamma_f_vec(fs_points[0], fs_points[0],gamma_f_orbit_arr))
# test_arr = tranform_a_vec(a2_arr,a3_arr,fs_points[0]).conj() * tranform_a_vec(a1_arr,a4_arr,fs_points[0])
# np.save("test_arr.npy",test_arr)
# np.save("flip_V_sum.npy",flip_V_sum)
# sys.exit()
# def plot_test_gamma(gamma_data, path, name):
#     num_s = 130*(32/kn)
#     ax = plt.subplot(111)
#     ax.set_aspect('equal', adjustable='box')
#     ax.set_xlabel(r'$k_{x}/\pi$')
#     ax.set_ylabel(r'$k_{y}/\pi$')
#     ax.set_xlim(-0.75*multi,0.75*multi)
#     ax.set_ylim(-0.75*multi,0.75*multi)
#     second_bz = plt.Polygon(kagome_lattice, fill=None, edgecolor='black')
#     ax.add_patch(second_bz)
#     first_bz = plt.Polygon(kagome_lattice_bz, fill=None, edgecolor='black')
#     ax.add_patch(first_bz)
#     ff=ax.scatter(kx_point_phi,ky_point_phi, c=gamma_data, alpha=0.5, cmap='rainbow',s=num_s,marker='.')
#     plt.colorbar(ff)
#     fig1 = plt.gcf()  #gcf: Get Current Figure
#     fig1.savefig(path+"//"+name+".png", dpi=300)

# for a1 in range(band_number):
#         for a2 in range(band_number):
#             for a3 in range(band_number):
#                 for a4 in range(band_number):
#                     plot_test_gamma(test_gamma_arr[a1,a2,a3,a4],path_phi_fig,"gamma_"+str(a1)+str(a2)+str(a3)+str(a4))

# for a1 in range(band_number):
#         for a2 in range(band_number):
#             for a3 in range(band_number):
#                 for a4 in range(band_number):
#                     plot_test_gamma(flip_V_sum[a1*band_number+a2,a3*band_number+a4].real,path_phi_fig,"V_s_"+str(a1)+str(a2)+str(a3)+str(a4))

# sys.exit()
gamma_matrix1 = gamma_f_vec(fs_points.reshape(1,fs_points_number,2),fs_points.reshape(fs_points_number,1,2),gamma_f_orbit_arr)
gamma_matrix2 = gamma_f_vec(fs_points.reshape(1,fs_points_number,2),-fs_points.reshape(fs_points_number,1,2),gamma_f_orbit_arr)
gamma_matrix = (gamma_matrix1+gamma_matrix2)/2
np.save("gamma_matrix.npy",gamma_matrix)
print_and_save_txt("gamma_matrix.shape",gamma_matrix.shape)

matrix_arr = - fermi_velocity * gamma_matrix * (1/(4*pi**2))
np.save("matrix_arr.npy",matrix_arr)
print(matrix_arr.dtype)
vals, vecs = eigs(matrix_arr, k=para.number_eig, which = 'LR')
print_and_save_txt(vals)
np.save(SOURCE+"//"+name_time+"//"+"phi.npy",vecs)
for vec_num in range(para.number_eig):
    plot_fs_point_phi(vecs,path=path_model_fig,name="fs_new"+str(vec_num),number_s=3,vec_num=vec_num)

elapsed3 = f(time.time() - start3)
print_and_save_txt("fs Time used:",elapsed3)



# #B-S本征方程计算---begin---
# start3 = time.time()
# def func_bs(phi_f):
#     return phi_value(greenrme,V_bs_me,phi_f)
# sum_num = band_number*band_number*kn*kn
# opera_bs = LinearOperator((sum_num,sum_num), matvec=func_bs,dtype=np.float32)

# vals, vecs = eigs(opera_bs, k=para.number_eig, which = 'LR')
# print_and_save_txt(vals)
# np.save(phi_data_str(),vecs)
# elapsed3 = f(time.time() - start3)
# print_and_save_txt("B-S eq Time used:",elapsed3)

# vecs.shape = band_number,band_number,kn,kn,para.number_eig  
# for num_phi in range(para.number_eig):
#     plot_phi(vecs[:,:,:,:,num_phi],path_phi_fig,s(num_phi)+"_phi_orbit")
#     plot_phi_band(vecs[:,:,:,:,num_phi],path_phi_fig, np.real, s(num_phi)+"_phi_band")







'''
def plot_phi_band(phi_data, path, func, name):
    "phi_data:o,o,kn,kn"
    num_s = 130*(0.5/kn)
    #transf_mat = band_ma(kx_band, ky_band)
    #transf_mat_mu = band_ma(-kx_band, -ky_band)
    transf_mat = band_vec(kx_band, ky_band)[1]
    transf_mat_mu = band_vec(-kx_band, -ky_band)[1]
    transf_mat_conj = np.conj(transf_mat.transpose(0,1,3,2))
    phi_data_0 = phi_data.transpose(2,3,0,1)
    phi_plot_data = transf_mat@phi_data_0@transf_mat_mu
    for i  in range(band_number):
        for j in range(band_number):
            #plt.figure(figsize=(10,10))
            plt.xlabel(r'$k_{x}/\pi$')
            plt.ylabel(r'$k_{y}/\pi$')
            f=plt.scatter(kx_point_phi,ky_point_phi,c=func(phi_plot_data[:,:,i,j]), cmap='rainbow',s=num_s, alpha=1)
            #f=plt.imshow(np.real(phi_last[i,j,int(Tn/2),:,:,int(kn/2)]),extent =extent, cmap='rainbow', clim=(-0.065,0.065))
            plt.colorbar(f)
            #plt.xlim(-0.75*multi,0.75*multi)
            #plt.ylim(-0.75*multi,0.75*multi)
            fig1 = plt.gcf() #gcf: Get Current Figure
            fig1.savefig(path+"//"+name+"_"+s(i)+s(j)+".png", dpi=800)
            plt.close()

for num_phi in range(para.number_eig):
    #plot_phi(vecs[:,:,:,:,num_phi],path_phi_fig,s(num_phi)+"_phi_orbit")
    plot_phi_band(vecs[:,:,:,:,num_phi],path_phi_fig, np.real, s(num_phi)+"_phi_band")
'''