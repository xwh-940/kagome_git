from code_func import *

#对称性限制
def symmetry_singlet(array2):
    array2_re = tr.flip(array2, [0,1])
    array3 = (array2_re + array2)/2
    return array3

def symmetry_singlet_back(array2):
    array2_re = tr.flip(array2, [0])
    array3 = (array2_re + array2)/2
    array3_re = tr.flip(array3, [1])
    array4 = (array3_re + array3)/2
    return array4

def symmetry_triplet(array2):
    array2_re = tr.flip(array2, [0,1])
    array3 = (-array2_re + array2)/2
    return array3

def symmetry_dx2y2(array0):
    a0 = array0.transpose(1,0)
    array1 = (-a0 + array0)/2
    a1 = tr.flip(tr.flip(array1,[0]).transpose(1,0),[0])
    array2 = (-a1 + array1)/2

    array2_re = tr.flip(array2, [0])
    array3 = (array2_re + array2)/2
    array3_re = tr.flip(array3, [1])
    array4 = (array3_re + array3)/2
    return array4

def symmetry_s(array0):
    a0 = array0.transpose(2-1,1-1)
    array1 = (a0 + array0)/2
    a1 = tr.flip(tr.flip(array1,[1-1]).transpose(2-1,1-1),[1-1])
    array2 = (a1 + array1)/2

    array2_re = tr.flip(array2, [1-1])
    array3 = (array2_re + array2)/2
    array3_re = tr.flip(array3, [2-1])
    array4 = (array3_re + array3)/2
    return array4

def symmetry_dxy(array0):
    a0 = array0.transpose(2-1,1-1)
    array1 = (a0 + array0)/2
    a1 = tr.flip(tr.flip(array1,[1-1]).transpose(2-1,1-1),[1-1])
    array2 = (a1 + array1)/2

    array2_re = tr.flip(array2, [1-1])
    array3 = (-array2_re + array2)/2
    array3_re = tr.flip(array3, [2-1])
    array4 = (-array3_re + array3)/2
    return array4

def symmetry_dxy_anti(array0):
    a0 = array0.transpose(2-1,1-1)
    array1 = (-a0 + array0)/2
    a1 = tr.flip(tr.flip(array1,[1-1]).transpose(2-1,1-1),[1-1])
    array2 = (-a1 + array1)/2

    array2_re = tr.flip(array2, [1-1])
    array3 = (-array2_re + array2)/2
    array3_re = tr.flip(array3, [2-1])
    array4 = (-array3_re + array3)/2
    return array4

def symmetry_chance(array,a2,a3):
    if para.symm == "singlet":
        return symmetry_singlet(array)
    if para.symm == "triplet":
        return symmetry_triplet(array)
    if para.symm == "dxy":
        return symmetry_dxy(array)
    if para.symm == "dx2y2":
        return symmetry_dx2y2(array)
    if para.symm == "s":
        return symmetry_s(array)
    if para.symm == "no_symm":
        return array
    if para.symm == "dxy_dx2y2":
        if a2 == 0 and a3 == 0:
            return symmetry_dxy(array)
        else:
            return symmetry_dx2y2(array)
    if para.symm == "dx2y2_dxy":
        if a2 == 0 and a3 == 0:
            return symmetry_dx2y2(array)
        elif a2 == 1 and a3 == 1:
            return symmetry_dxy(array)
        else:
            return array
    if para.symm == "s+-":
        if a2 == 0 and a3 == 0:
            return symmetry_s(array)
        elif a2 == 1 and a3 == 1:
            return -symmetry_s(array)
        else:
            return array


def distance(arr):
    'input:n*2 ; output:n*2'
    arr_one = np.array([0,0]).reshape(1,2)
    arr0 = np.concatenate((arr_one,arr))
    arr1 = np.concatenate((arr,arr_one))
    arr_delta = (arr1-arr0)[1:-1]
    arr_distance = np.concatenate((np.array([0]),norm(arr_delta,axis=1)))
    arr_distance_sum = np.cumsum(arr_distance)
    return arr_distance_sum

def tranfrom_90_to_60_back(arr_90):
    kx_90 = arr_90[0]
    ky_90 = arr_90[1]
    kx_60 = 2 * kx_90 / sqrt(3)
    ky_60 = ky_90 + kx_90 / sqrt(3)
    return np.array([kx_60, ky_60])

def tranfrom_90_to_60(arr_90):
    kx_90 = arr_90[0]
    ky_90 = arr_90[1]
    kx_60 = kx_90 / sqrt(3) - ky_90
    ky_60 = kx_90 / sqrt(3) + ky_90
    return np.array([kx_60, ky_60])
tran_96_vec = vec(tranfrom_90_to_60, signature='(n)->(n)')

def period(num, period_begin, period_end):
    'period_end > period_begin'
    period_length = period_end - period_begin
    if num > period_end:
        period_num = num - period_length
    if num < period_begin:
        period_num = num + period_length
    else:
        period_num = num
    return period_num
period_vec = vec(period,signature='(),(),()->()')

def flip_f_back(arr):
    'arr:n * n'
    len_num = arr.shape[0]
    arr_0 = arr.reshape(2,len_num//2,2,len_num//2)
    arr_1 = np.flip(arr_0,axis=(0,2))
    arr_2 = arr_1.reshape(len_num,len_num)
    return arr_2

def flip_f(arr):
    'arr:n * n'
    len_num = arr.shape[0]
    return np.roll(arr,shift=(len_num//2-1,len_num//2-1),axis=(0,1))
flip_f_vec = vec(flip_f,signature='(n,n)->(n,n)')


