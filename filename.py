from code_func import *

def green0m_str(a1,a2):
    "str of green0m"
    return data+"//"+"green0m_"+a1+a2+".npy"

def green0_inversem_str(a1,a2):
    "str of green0_inversem"
    return data+"//"+"green0_inversem_"+a1+a2+".npy"

chi0_matrix_str = "chi0m.npy"
chis_str = "chis.npy"

@njit
def greenr(a1,a2): 
    "a1,a2,n is str;print str"
    return data+"//"+"greenr"+a1+a2+"_"+".npy"

@njit
def chir():
    "n is str;print str"
    return data+"//"+"chir"+"_"+".npy"

def V_bs_fft_str(a1,a2,a3,a4):
    "str of Vs1"
    return data+"//"+"V_bs_"+a1+a2+a3+a4+".pt"

def phi_str(a1,a2): 
    "a1,a2,n is str;print str"
    return data+"//"+"phi"+a1+a2+"_"+".pt"

def phi_data_str(): 
    return SOURCE+"//"+name_time+"//"+"phi"+".npy"

