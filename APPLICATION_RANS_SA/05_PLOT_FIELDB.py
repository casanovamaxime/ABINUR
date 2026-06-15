from utils.helpers_plots import * 
from utils.helpers_operators import *
from utils.helpers_read_write_qoi import * 
from utils.helpers_read_hf import *  
import matplotlib
import numpy as np
import scipy
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from matplotlib.ticker import FormatStrFormatter
matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30) 
continue_annotation()

print("=====START=====")

#GENERAL COMMANDE
#if false do not perform DA
DA_TRUE = False
#load posterior 'A' or prior 'B'
STATE = 'B'
#if false start from the last Reynolds (12600)
RESTART_0 = False

#REYNOLDS
Re_s = Reynolds(RESTART_0)

#PARALLELISM
comm = mpi.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print(rank,size)

#MESH AND FESPACE
dX = dx(metadata={"quadrature_degree":3*1+1})
set_annotation = True
#Define mesh
mesh,h,n = define_mesh()
#Define function space
P0,P1,P1B,Q = define_fe_space(mesh)
#Define functions and test functions
qx,ux,px,nutx,phiux,phipx,phinutx = define_qx_phiqx(Q)
#Read prior or posterior state
qx = read_qx(Q,RESTART_0,STATE)
ux,px,nutx = split(qx)
#Local space variables
x, y = SpatialCoordinate(mesh)
xes,yes = return_coord_cdc(mesh,P1)

#HIGH FIDELITY OBSERVATION
#Smooth observations
uvhfx,nuthfx = read_field_HF(P1,P1B)
#Add normally distributed noise
pcg = PCG64(seed=1)
rg = RandomGenerator(pcg)
sigma_y = 0.01
etax = rg.normal(P1B, 0., sigma_y)
uvhfx_noise = define_noisy_HF(etax,P1,P1B)
#Observation domain for global error
Omega_y_full=And(And(x>1,y<1.25),And(x<11,y<1.25)) 

#FOR STORING FIELD
field=Function(P1)
ff=Function(P1B)
fielddns=Function(P1)
errdns = Function(P1B)


#ERR B
errdns.dat.data[:,0] = abs((qx.dat.data[0][:,0] - uvhfx.dat.data[:,0]))
fielddns.interpolate((errdns[0]))
plot_field(fielddns,fielddns,xes,yes,mesh,'errB',0.6,0,16)

#SIGFB
with open('data_UQB/sigma_fB_.txt') as f:
    linesfx = f.readlines()     
    for it,el in enumerate(linesfx):
        field.dat.data[it]=linesfx[it] 
field.interpolate(field)
plot_field(field,field,xes,yes,mesh,'sigfB',1.9,0,32)

#SIGFUB
with open('data_UQB/sigma_uB_.txt') as f:
    linesfx = f.readlines()     
    for it,el in enumerate(linesfx):
        ff.dat.data[it,0]=linesfx[it] 
field.interpolate(ff[0])
plot_field(field,field,xes,yes,mesh,'sigB',0.6,0,16)
