from parameters import *

bz_l=1
bz_kn=kn
bz_k_orgin = np.linspace(0,2,bz_kn+1)*multi*(1/np.sqrt(3))
#(2/sqrt(3))
bz_kx_list = bz_k_orgin[:-1]
bz_ky_list = bz_k_orgin[:-1]


bz_b1 = np.array([0,2/sqrt(3)])*multi
bz_b2 = np.array([1,-1/sqrt(3)])*multi
bz_b3 = np.array([-1,-1/sqrt(3)])*multi
bz_b4 = np.array([-2,0])*multi
bz_point_arr = np.zeros((bz_kn,bz_kn,2))

point = [[Point(np.sqrt(3)/2*kx+np.sqrt(3)/2*ky,1/2*ky-1/2*kx) for ky in bz_ky_list] for kx in bz_kx_list]
poly_context = {'type': 'MULTIPOLYGON',
    'coordinates': [[[[2/3*multi, 0], [1/3*multi, 1/sqrt(3)*multi], [-1/3*multi,1/sqrt(3)*multi], [-2/3*multi,0],[-1/3*multi,-1/sqrt(3)*multi],[1/3*multi,-1/sqrt(3)*multi]]]]}
poly_shape = asShape(poly_context)
logic_list=[[poly_shape.intersects(point[i][j]) for j in range(bz_kn)] for i in range(bz_kn)]

for ikx in range(bz_kn):
    for iky in range(bz_kn):
        kx_value = bz_kx_list[ikx]
        ky_value = bz_ky_list[iky]
        point_2 = np.array([np.sqrt(3)/2*kx_value+np.sqrt(3)/2*ky_value, 1/2*ky_value-1/2*kx_value])
        if logic_list[ikx][iky]==True:
            bz_point_arr[ikx,iky] = point_2
        else:
            b10 = point_2 + bz_b1
            b11 = point_2 - bz_b1
            b20 = point_2 + bz_b2
            b21 = point_2 - bz_b2
            b30 = point_2 + bz_b3
            b31 = point_2 - bz_b3
            b40 = point_2 + bz_b4
            b41 = point_2 - bz_b4

            if poly_shape.intersects(Point(b10[0], b10[1]))==True:
                bz_point_arr[ikx,iky] = b10
            elif poly_shape.intersects(Point(b11[0], b11[1]))==True:
                bz_point_arr[ikx,iky] = b11
            elif poly_shape.intersects(Point(b20[0], b20[1]))==True:
                bz_point_arr[ikx,iky] = b20
            elif poly_shape.intersects(Point(b21[0], b21[1]))==True:
                bz_point_arr[ikx,iky] = b21
            elif poly_shape.intersects(Point(b30[0], b30[1]))==True:
                bz_point_arr[ikx,iky] = b30
            elif poly_shape.intersects(Point(b31[0], b31[1]))==True:
                bz_point_arr[ikx,iky] = b31
            elif poly_shape.intersects(Point(b40[0], b40[1]))==True:
                bz_point_arr[ikx,iky] = b40
            elif poly_shape.intersects(Point(b41[0], b41[1]))==True:
                bz_point_arr[ikx,iky] = b41
            else:
                #print("what?")
                bz_point_arr[ikx,iky] = np.nan

kx_point = (bz_point_arr[:,:,0])
ky_point = (bz_point_arr[:,:,1])

bz_k_orgin = np.linspace(-1,1,bz_kn+1)*multi*(1/np.sqrt(3))
#(2/sqrt(3))
bz_kx_list = (bz_k_orgin[1:]+bz_k_orgin[:-1])/2
bz_ky_list = (bz_k_orgin[1:]+bz_k_orgin[:-1])/2
#bz_kx_list = k_orgin[:-1]
#bz_ky_list = k_orgin[:-1]


bz_b1 = np.array([0,2/sqrt(3)])*multi
bz_b2 = np.array([1,-1/sqrt(3)])*multi
bz_b3 = np.array([-1,-1/sqrt(3)])*multi
bz_point_arr = np.zeros((bz_kn,bz_kn,2))

point = [[Point(np.sqrt(3)/2*kx+np.sqrt(3)/2*ky,1/2*ky-1/2*kx) for ky in bz_ky_list] for kx in bz_kx_list]
poly_context = {'type': 'MULTIPOLYGON',
    'coordinates': [[[[2/3*multi, 0], [1/3*multi, 1/sqrt(3)*multi], [-1/3*multi,1/sqrt(3)*multi], [-2/3*multi,0],[-1/3*multi,-1/sqrt(3)*multi],[1/3*multi,-1/sqrt(3)*multi]]]]}
poly_shape = asShape(poly_context)
logic_list=[[poly_shape.intersects(point[i][j]) for j in range(bz_kn)] for i in range(bz_kn)]

for ikx in range(bz_kn):
    for iky in range(bz_kn):
        ky_value = bz_kx_list[ikx]
        kx_value = bz_ky_list[iky]
        point_2 = np.array([np.sqrt(3)/2*kx_value+np.sqrt(3)/2*ky_value, 1/2*ky_value-1/2*kx_value])
        if logic_list[ikx][iky]==True:
            bz_point_arr[ikx,iky] = point_2
        else:
            b10 = point_2 + bz_b1
            b11 = point_2 - bz_b1
            b20 = point_2 + bz_b2
            b21 = point_2 - bz_b2
            b30 = point_2 + bz_b3
            b31 = point_2 - bz_b3
            if poly_shape.intersects(Point(b10[0], b10[1]))==True:
                bz_point_arr[ikx,iky] = b10
            elif poly_shape.intersects(Point(b11[0], b11[1]))==True:
                bz_point_arr[ikx,iky] = b11
            elif poly_shape.intersects(Point(b20[0], b20[1]))==True:
                bz_point_arr[ikx,iky] = b20
            elif poly_shape.intersects(Point(b21[0], b21[1]))==True:
                bz_point_arr[ikx,iky] = b21
            elif poly_shape.intersects(Point(b30[0], b30[1]))==True:
                bz_point_arr[ikx,iky] = b30
            elif poly_shape.intersects(Point(b31[0], b31[1]))==True:
                bz_point_arr[ikx,iky] = b31
            else:
                #print("what?")
                bz_point_arr[ikx,iky] = np.nan
kx_point_phi = (bz_point_arr[:,:,0])
ky_point_phi = (bz_point_arr[:,:,1])
