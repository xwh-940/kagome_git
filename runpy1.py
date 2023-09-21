import numpy as np
import sys
import os


def array(arg):
    if not arg.__class__ == np.ndarray:
        return np.array([arg])
    return np.array(arg)

def s(arg):
    return " " + str(arg)


def run(mu=-0.2, u0=1.0):
    for i1 in array(mu):
        for i2 in array(u0):
            os.system                          \
            (                                  \
                "python para_code.py"      \
                + s(i1) + s(i2)    \
                + s(kn) + s(Tn) + s(T)  \
                + s(test) +  s(channel) + s(result_name)\
            )


#mu_para  = np.array([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
mu_para  = np.array([-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9])
u0_para  = np.array([])
u1_para  = np.array([-0.2, -0.1, 0.1])
u11_para = np.array([])
j_para   = np.array([])
j1_para  = np.array([])

#-----------------------------------------------------
#-----------------------------------------------------
test = 0
"0 is test ; 1 is true compute"
assert test == 0 or test == 1 , "test must be 0 or 1"
#-----------------------------------------------------
#-----------------------------------------------------
if test == 0:
    "test"
    result_name = 'rpa_test'
    channel = "singlet"
    kn = 128
    Tn = 1024
    # T = 0.05
    T = 0.02 
    #self_consistent_number = 3
    #iteration_number = 100
    #number_eig = 2
    #symm = "dx2y2"      ; assert_symm()
    #mod_orbit = 0       ; assert_mod_orbit()
    #anti = 0      # 0：normal_dxy 1:anti_dxy

    print('begin')
    # run(mu=0.05,u0=2.5)
    # run(mu=0.05,u0=2.0)
    # run(mu=2.2,u0=1.5)
    # run(mu=0.3,u0=3.0)

if test == 1:
    "true compute"
    result_name = 'rpa_test'
    channel = "singlet"
    kn = 128
    Tn = 1024
    T = 0.05
    #self_consistent_number = 20
    #iteration_number = 50
    #number_eig = 2
    #mod_orbit = 0       ; assert_mod_orbit()
    #anti = 0      # 0：normal_dxy 1:anti_dxy


    
    #mu_para  = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])*2+0.05*2    
    mu_para=np.arange(-0.6,1.8,0.1)
    #symm = "dx2y2"    ; assert_symm()   
    
    
    run(mu=-0.0,u0=1.0)
    #run(mu=mu_para,u0=1.0)

        
    
          
    

print("run end")




