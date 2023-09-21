from susceptibility import *


def chi0r(a1,a2,a3,a4,n):
    #print("chi0r time0",time.process_time())
    "chi0r is = susceptibity of renormalization"
    if n ==0:
        return chi0f(a1,a2,a3,a4)
    g0_r1 = fft.fftn(np.load(greenr(s(a3),s(a1))))
    g0_ir2 = fft.ifftn(np.load(greenr(s(a2),s(a4))))
    chi00_r = ne("-(g0_r1 * g0_ir2)*Tn*T").astype(np.complex64)
    chi0r0  = fft.ifftn(chi00_r)
    chi0r0 = np.real(chi0r0)
    #print("chi0r time1",time.process_time())
    return chi0r0

def transf42n(func,b1,b2,n):
    "2*2*2*2 -> 4*4 "
    num = band_number
    return func(b1//num,b1%num,b2//num,b2%num,n)

def transf24n(func,a1,a2,a3,a4,n):
    "4*4 -> 2*2*2*2 "
    num = band_number
    return func(num*a1+a2,num*a3+a4,n)

def chi0r_m(n):
    "(4,4,Tn,kn,kn) all of renormalizable chi0"
    #print("chi0r_m",time.process_time())
    if n==0:
        return np.load(data+"//"+"chi0m.npy")
    chi0r_m0 = np.array([[transf42n(chi0r,i,j,n) for j in range(band_number**2)] for i in range(band_number**2)])
    return chi0r_m0

@njit
def V_r(wn0,kx0,ky0,n,arr):
    "effective interaction with given wn,kx,ky,kz"
    chi = con(arr[:,:,wn0,kx0,ky0])
    V_r0 = v15 * Us @ (inv(id4 - chi @ Us) - id4) @ chi @ Us\
         + v05 * Uc @ (inv(id4 + chi @ Uc) - id4) @ chi @ Uc\
         + v05 * (Us @ chi @ Us + Uc @ chi @ Uc)
    #print("V_r time0",time.process_time())
    return V_r0

@njit
def V_r1(n,arr):
    "effective interaction with given orbit"
    V_r10 = np.empty((Tn, kn, kn, band_number_square, band_number_square), dtype=np.complex64)
    for para in prange(para_array.shape[0]):
        V_r10[para_array[para,0],para_array[para,1],para_array[para,2]] \
            = np.real(V_r(para_array[para,0],para_array[para,1],para_array[para,2],n,arr))
    return  V_r10

def value(n,arr,greenf):
    def sigma(a3,a1,n):
        "self-energy with given orbit"
        sigma0 = np.zeros((Tn, kn, kn), dtype=np.complex64)
        for a2 in prange(band_number):
            for a4 in prange(band_number):
                V_r1_r = fft.fftn(V_r1_copy[:,:,:,band_number*a1+a2,band_number*a3+a4])
                green_r = fft.fftn(greenf[a2,a4])
                sigma_r0 = ne("( V_r1_r * green_r)*(T/(kn*kn))").astype(np.complex64)
                sigma0 += fft.ifftn(sigma_r0)    
        return sigma0

    def green_inverse(a1,a2,n):
        #Dyson equation
        green_inverse0 = green0_inverseme(a1,a2) - sigma(a1,a2,n)
        return green_inverse0
    if n == 0:
        for orbit in prange(orbitral_array.shape[0]):
            np.save(greenr(s(orbitral_array[orbit,0]),s(orbitral_array[orbit,1]))\
                ,green0me(orbitral_array[orbit,0],orbitral_array[orbit,1]))
    else:
        V_r1_copy = V_r1(n,arr)
        green_inv_arr = con(np.array([[green_inverse(i,j,n-1) for j in range(band_number)] for i in range(band_number)]).transpose((2,3,4,0,1)))
        greenr_arr = con(inv(green_inv_arr).transpose((3,4,0,1,2)))
        for orbit in prange(orbitral_array.shape[0]):
            np.save(greenr(s(orbitral_array[orbit,0]),s(orbitral_array[orbit,1]))\
                ,greenr_arr[orbitral_array[orbit,0],orbitral_array[orbit,1]])
    return 0


