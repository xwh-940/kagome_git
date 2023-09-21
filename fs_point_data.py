from model_1_kagome import *

def band_vals_fs(k_90):
    k_60 = tran_96_vec(k_90)
    return band_value(k_60[0]*np.pi,k_60[1]*np.pi)
band_vals_fs_vec = vec(band_vals_fs, signature='(m)->(n)')

extreme_point = np.array([[2/3, 0], [1/3, 1/sqrt(3)], [-1/3,1/sqrt(3)], [-2/3,0],[-1/3,-1/sqrt(3)],[1/3,-1/sqrt(3)]])
extreme_point_axis  = extreme_point/2 

center_point = np.array([0,0])
center_point_axis  = np.array([0, sqrt(3)/3])


def divide_segment(k):
    "(2)->(128,2)"
    t = np.linspace(0, 1, num=128)
    x, y = t * k[0], t * k[1]
    arr = np.zeros((128,2))
    arr[:,0], arr[:,1] = x, y
    return arr
divide_segment_vec = vec(divide_segment,signature='(n)->(m,n)')

def extreme_or_center(band_vals_fs):
    extreme_k = divide_segment(extreme_point_axis[0])+extreme_point[0]
    center_k = divide_segment(center_point_axis)+center_point
    extreme_min = np.abs(band_vals_fs(extreme_k)).min(axis=0)
    center_min = np.abs(band_vals_fs(center_k)).min(axis=0)

    if extreme_min.min() < center_min.min():
        #print("extreme:", extreme_min.min())
        extreme_k_arr = divide_segment_vec(extreme_point_axis).reshape(6,128,2)+extreme_point.reshape(6,1,2)
        arr0 = np.abs(band_vals_fs(extreme_k_arr))[:,:,extreme_min.argmin()]
        k0 = extreme_k_arr[np.arange(6),arr0.argmin(axis=1)]
        return "extreme", extreme_min.argmin(), k0
    else:
        #print("center:", center_min.min())
        arr0 = np.abs(band_vals_fs(center_k))[:,center_min.argmin()]
        k0 = center_k[arr0.argmin()]
        return "center", center_min.argmin(), k0

def fs_points_func(rational):
    N_num = 128
    dl = 2*np.pi/N_num
    radius = dl*rational
    def init_root(func_ek):
        return extreme_or_center(func_ek)   
    init_root_result = init_root(band_vals_fs_vec)
    which_center = init_root_result[0]
    which_band = init_root_result[1]
    first_point = init_root_result[2]

    def next_point(init_point,init_index):
        number_theta = 50
        theta = np.linspace(0,2*np.pi,number_theta)
        other_ordinary = radius * np.array([np.sin(theta), np.cos(theta)]).transpose(1,0)
        absolute_ordinary = init_point + other_ordinary
        def min_set(arr,last_index):
            'arr:(50,2)'
            indexs_arr = (last_index + np.arange(int((-number_theta-10)/4),int((number_theta+10)/4),1))%number_theta
            indexs=np.abs(band_vals_fs_vec(arr)[indexs_arr,which_band]).argmin()
            return arr[indexs_arr[indexs]], indexs_arr[indexs]
        return min_set(absolute_ordinary,init_index)

    def distance_of_first(first_point,last_point):
        delta = last_point - first_point
        # mod = np.sqrt(delta[0]**2+delta[1]**2-2*delta[0]*delta[1]*np.cos(pi/3))
        mod = np.sqrt(delta[0]**2+delta[1]**2)
        return mod
    if which_center == 'center':
        ll = 5000
        fermi_S = np.zeros((ll,2))
        indexs_arr = np.zeros((ll),dtype=np.int32)
        distance_arr = np.zeros((ll),dtype=np.float64)
        for num in range(ll):
            if num == 0:
                fermi_S[num] = first_point
            else:
                _inter = next_point(fermi_S[num-1],indexs_arr[num-1])
                # fermi_S[num] = _inter
                fermi_S[num] = _inter[0]
                indexs_arr[num] = _inter[1]
                distance_arr[num] = distance_of_first(first_point,fermi_S[num])
                if distance_arr[num] < 5*radius and num > 5:
                    if distance_arr[num] > distance_arr[num-1]:
                        last_number = num-1
                        # print("number of points: ", last_number)
                        break
        fs_points = fermi_S[:last_number]

    if which_center == 'extreme':
        ll = 5000
        fermi_S = np.zeros((6,ll,2))
        last_number = 0
        num_arr = np.zeros((6),dtype=np.int32)
        for point_nums in range(6):
            indexs_arr = np.zeros((ll),dtype=np.int32)
            distance_arr = np.zeros((ll),dtype=np.float64)
            for num in range(ll):
                if num == 0:
                    fermi_S[point_nums,num] = first_point[point_nums]
                else:
                    _inter = next_point(fermi_S[point_nums,num-1],indexs_arr[num-1])
                    fermi_S[point_nums,num] = _inter[0]
                    indexs_arr[num] = _inter[1]
                    distance_arr[num] = distance_of_first(first_point[point_nums],fermi_S[point_nums,num])
                    if distance_arr[num] < 5*radius and num > 5:
                        if distance_arr[num] > distance_arr[num-1]:
                            last_number += num-1
                            num_arr[point_nums] = num-1
                            # print("number of points: ", last_number)
                            break
        # print("number of points: ", last_number)
        fs_points = []
        for point_nums in range(6):
            fs_points += list(fermi_S[point_nums,:num_arr[point_nums]])
        fs_points = np.array(fs_points)
    return fs_points, last_number, which_center, which_band

fs_numbers = 256
def fs_points_func_number(rational):
    return fs_points_func(rational)[1] - fs_numbers


# dl_target = optimize.bisect(fs_points_func_number,0.2, 0.5)
dl_target = optimize.bisect(fs_points_func_number,0.2, 1.0)

_inter_set = fs_points_func(dl_target)
fs_points = _inter_set[0]
which_center = _inter_set[2]
which_band = _inter_set[3]


print_and_save_txt("dl:",dl_target)
print_and_save_txt("shape of fs point:", fs_points.shape)
print_and_save_txt("center of fs:", which_center)
print_and_save_txt("band of fs:", which_band)
print_and_save_txt("---------------------")

fs_num = fs_points.shape[0]
fs_points_90 = fs_points.copy()
fs_points = tran_96_vec(fs_points)
# fs_points_90 = fs_points[:,0].reshape(fs_num,1)*kx_basis.reshape(1,2)+fs_points[:,1].reshape(fs_num,1)*ky_basis.reshape(1,2)
np.save("fs_points_90.npy",fs_points_90)
np.save("fs_points_60.npy",fs_points)
# phi = np.load("phi.npy")




def plot_fs_point(path,name,number_s):
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
    ax.scatter(fs_points_90[:,0],fs_points_90[:,1],s=number_s)
    # ax.scatter(fs_points_90[:,0],fs_points_90[:,1],s=number_s,c=phi[:,vec_num].real,cmap='rainbow')
    
    fig1 = plt.gcf() #gcf: Get Current Figure
    fig1.savefig(path+"//"+name+".png", dpi=800)
    plt.close()

np.save(SOURCE+"//"+name_time+"//"+"fs_points.npy",fs_points_90)
