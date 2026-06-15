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
   
#MESH AND FESPACE
dX = dx(metadata={"quadrature_degree":3*1+1})
set_annotation = True
#Define mesh
mesh,h,n = define_mesh()
#Define function space
P0,P1,P1B,Q = define_fe_space(mesh)
#Local space variables
x, y = SpatialCoordinate(mesh)
xes,yes = return_coord_cdc(mesh,P1)
#Color
cmap = plt.get_cmap('RdBu_r')
blue = cmap(0.2)     # low end (blue)
orange = cmap(0.75)   # high end (orange/red)

#READ VARIANCE
varpx = Function(P1)
with open('data_UQB/sigma_pA_.txt') as f:
    linesfx = f.readlines()     
for it,el in enumerate(linesfx):
    varpx.dat.data[it]=float(linesfx[it])**2  
#REMOVE OBS. PART FROM HESSIAN FULL
with open('data_UQA_FULL/sigma_pA_.txt') as f:
    linesfx = f.readlines()     
for it,el in enumerate(linesfx):
    varpx.dat.data[it]-=float(linesfx[it])**2    
varpx.dat.data[:]=np.sqrt(abs(varpx.dat.data[:]))    
            
#READ CP
xcfles,xcples,cples,cfles = read_cp_cf_LES()
cp=[]
pinf=0.318
with open('data_SA_DA/CpA.txt') as f:
    lines = f.readlines()
for it,el in enumerate(lines):
    cp.append(-(float(el))+pinf)
varCp=[]
for i in range(len(xes)):
    varCp.append((2*varpx.at(xes[i],yes[i]))) 
 
#PLOT     
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5,5))
title = ''
xlabel = r'$x$'
ylabel = r'$C_p$'
ax.plot(xcples, cples,color='k',zorder=2, ls='-',linewidth=2)
ax.plot(xes, cp,color=blue,zorder=3, ls='-.',linewidth=3)
ax.fill_between(xes,np.array(cp)+np.array(varCp), np.array(cp)-np.array(varCp), zorder=0,edgecolors='0.',facecolor=blue, hatch='/',alpha=0.25) 
ax.set_title(title)
plt.setp(ax.spines.values(), lw=2)
ax.set_xlabel(xlabel,fontsize=30)
ax.set_ylabel(ylabel,fontsize=30)
#ax.invert_yaxis()
fig.tight_layout()
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
plt.minorticks_on()
plt.subplots_adjust(left=0.3)
plt.xlim(1, 10)
plt.ylim(-0.2, 1.6)
plt.savefig("cp.png")     
plt.show()


#READ CF
cf=[] 
with open('data_SA_DA/CfA.txt') as f:
    lines = f.readlines()
for it,el in enumerate(lines):
    cf.append(float(el))
varCf=[] 
varCf2=[] 
with open('data_UQB/sigma2_CfBA_.txt') as f:
    lines = f.readlines()
for it,el in enumerate(lines):
    varCf.append(float(el))
with open('data_UQA_FULL/sigma2_CfA.txt') as f:
    lines = f.readlines()
for it,el in enumerate(lines):
    varCf[it]-=(float(el))
for i in range(len(varCf)):
    varCf2.append((varCf[i])**0.5)
    
#PLOT    
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5,5))
title = ''
xlabel = r'$x$'
ylabel = r'$C_f$'
plt.hlines(0., 0, 10,color='k',linewidth=1)
ax.plot(xcfles, cfles,color='k',zorder=2, ls='-',linewidth=2)
ax.plot(xes, cf,color=blue,zorder=3, ls='-.',linewidth=3)
ax.fill_between(xes, np.array(cf)+(np.array(np.array(varCf2))), np.array(cf)-(np.array(np.array(varCf2))), zorder=0,edgecolors='0.',facecolor='skyblue', hatch='/',alpha=0.25)
ax.set_title(title)
plt.setp(ax.spines.values(), lw=2)
ax.set_xlabel(xlabel,fontsize=30)
ax.set_ylabel(ylabel,fontsize=30)
fig.tight_layout()
plt.xlim(1, 10)
plt.ylim(-0.0075, 0.025)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
plt.minorticks_on()
plt.subplots_adjust(left=0.3)
plt.savefig("cf.png")    
plt.show()
