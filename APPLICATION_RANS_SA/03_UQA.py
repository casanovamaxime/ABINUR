from utils.helpers_plots import * 
from utils.helpers_operators import *
from utils.helpers_read_write_qoi import * 
from utils.helpers_read_hf import *  
continue_annotation()

print("=====START=====")

#GENERAL COMMANDE
#if false do not perform DA
DA_TRUE = False
#load posterior 'A' or prior 'B'
STATE = 'A'
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
#0 sol
dnsx = read_qx(Q,RESTART_0,STATE)
dnsux,dnspx,dnsnutx = split(qx)
#Distance to the wall
Md = 0.005
d2x = distance2(P0,P1,Md)
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
uvhfx_noise = define_noisy_HF(etax,P1,P1B)#dnsux

#POINT DATA EVALUATION
#point observation
y_points = np.array([[7., 0.7],[7.5, 0.7],[8.,0.7],[8.5,0.7],[9.,0.7],[7., 0.5],[7.5, 0.5],[8.,0.5],[8.5,0.5],[9.,0.5],[7.5, 0.3],[8.,0.3],[8.5,0.3],[9.,0.3],[8.5,0.1],[9.,0.1] ]) 
#Compute dirac function 
hmin = float(mesh.cell_sizes.dat.data_ro.min())
delta_y = delta_y(x,y,y_points,P1,1*hmin)
#Observation domain for global error
Omega_y_full=And(And(x>1,y<1.25),And(x<11,y<1.25)) 


#DISCRETIZED PRIOR OPERATOR
sigma_f=0.75#8#0.8
lcor=0.25
kappa2_f = (1./(lcor**2))
tau_f = (1./(4*pi*kappa2_f*(sigma_f**2)))**0.5
#Kf matrix
Kf = define_Kf(P1,kappa2_f,tau_f)
#Mf full mass matrix
Mf = define_Mf(P1)
#Qf lumped cholesky matrix
Qf = define_Qf(P1)

#DA PARAM
#Read forcing
fx = read_fx(P1,RESTART_0,STATE) 
fx.interpolate(fx/Qf) 
#Optimizer parameters
f_iter = 0
f_opti_method = 'L-BFGS-B'
f_display = True
f_tol = 1e32

#UQ PARAM
#Number of eigenvalue to extract
rA = 30
#Check finite difference
check_grad_hess = False
number_of_dir_to_check = 5

#UQ A  
#Solve forward
qx = solve_SA_BorA(uvhfx,nuthfx,Qf,Re_s,qx,ux,px,nutx,phiux,phipx,phinutx,fx,Q,h,d2x,dX,set_annotation,STATE,DA_TRUE)
#Solve adjoint
fAx,Jhat = perform_DA(P1,ux,uvhfx_noise,fx,dX,tau_f,kappa2_f,f_opti_method,f_display,f_iter,f_tol,delta_y,sigma_y,DA_TRUE) 
#Solve uq
perform_UQ_A(xes,yes,delta_y,sigma_y,uvhfx,nuthfx,uvhfx_noise,Mf,Qf,Kf,Jhat,Re_s,qx,ux,px,nutx,phiux,phipx,phinutx,fAx,P1,P1B,Q,h,d2x,dX,rA,number_of_dir_to_check,check_grad_hess,set_annotation,STATE,DA_TRUE)

print("=====END OK=====")

