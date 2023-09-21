import sys
arg_ = sys.argv
mu = float(arg_[1])
u0 = float(arg_[2])
u1 = 0
u11 = 0
j = 0
j11 = 0
self_consistent_number = 5
iteration_number = 5

kn_fs = 512
kn = int(arg_[3])
Tn = int(arg_[4])
T = float(arg_[5])
number_eig = 7
test = int(arg_[6])
mod_orbit = 0
anti = 0
channel = arg_[7]
symm = channel
result_name = arg_[8]

import main

