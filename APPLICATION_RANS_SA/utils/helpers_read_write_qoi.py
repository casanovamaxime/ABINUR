from utils.helpers_uq import *

#define intermediate Reynolds to help convergence
def Reynolds(RESTART_0):
    if RESTART_0:
        Re_s =[1e0,1e1,3e1,1e2,3e2,6e2,1e3,1.25e3,1.5e3,2e3,2.25e3,2.5e3,3e3,4e3,5e3,6e3,7e3,8e3,9e3,1e4,12600.]
    else:
        Re_s = [12600.]
    return Re_s
    
#Write cost functional values                    
def write_j_(j_s):
    file = open('data_SA_DA/loss_.txt','w')
    for item in j_s:
        file.write(str(item)+"\n")

#Write forcing
def write_fx_(fx):
    file = open('data_SA_DA/fx'+'.txt','w')    
    for item in fx.dat.data[:]:
        file.write(str(item)+"\n") 
         
#Read forcing
def read_fx(P1,RESTART_0,name):
    fx=Function(P1)
    if RESTART_0 == False and name=='A':
        with open('data_SA_DA/fx'+'.txt') as f:
            linesfx = f.readlines()     
        for it,el in enumerate(linesfx):
            fx.dat.data[it]=linesfx[it]
    return fx
    
#Write state          
def write_qx(qx,name):
    file = open('data_SA_DA/qu'+name+'.txt','w')    
    for item in qx.dat.data[0][:,0]:
        file.write(str(item)+"\n")  
    file = open('data_SA_DA/qv'+name+'.txt','w')    
    for item in qx.dat.data[0][:,1]:
        file.write(str(item)+"\n") 
    file = open('data_SA_DA/qp'+name+'.txt','w')    
    for item in qx.dat.data[1][:]:
        file.write(str(item)+"\n") 
    file = open('data_SA_DA/qnut'+name+'.txt','w')    
    for item in qx.dat.data[2][:]:
        file.write(str(item)+"\n")  

#Read state
def read_qx(Q,RESTART_0,name):
    qx=Function(Q)
    if RESTART_0 == False:
        with open('data_SA_DA/qu'+name+'.txt') as f:
            linesfx = f.readlines()     
        for it,el in enumerate(linesfx):
            qx.dat.data[0][it,0]=linesfx[it]
        with open('data_SA_DA/qv'+name+'.txt') as f:
            linesfx = f.readlines()     
        for it,el in enumerate(linesfx):
            qx.dat.data[0][it,1]=linesfx[it]
        with open('data_SA_DA/qp'+name+'.txt') as f:
            linesfx = f.readlines()     
        for it,el in enumerate(linesfx):
            qx.dat.data[1][it]=linesfx[it]
        with open('data_SA_DA/qnut'+name+'.txt') as f:
            linesfx = f.readlines()     
        for it,el in enumerate(linesfx):
            qx.dat.data[2][it]=linesfx[it]       
    return qx

#Compute error norm
def compute_error_norm(ux,uvhfx,dx2,MASK_DOMAIN,name):
    sd = SubDomainData(MASK_DOMAIN)
    erru  = assemble(dot(ux[0]-uvhfx[0],ux[0]-uvhfx[0])*dx2(subdomain_data=sd))
    file = open('data_SA_DA/errnorm_'+name+'.txt','w')
    file.write("error L2 u :"+str(erru)+"\n")  

#Return coordinate of the channel
def return_coord_cdc(mesh,P1):
    coo = mesh.coordinates.dat.data
    idd=firedrake.cython.dmcommon.facet_closure_nodes(P1,[11])
    xe=[]
    ye=[]
    xes=[]
    yes=[]
    for i in idd:
        xe.append(coo[i][0])
        ye.append(coo[i][1])
    coo_e=np.zeros((len(xe),2))    
    coo_sorted_e=np.zeros((len(xe),2))
    for i in range(len(xe)):
        coo_e[i,0]=xe[i]  
        coo_e[i,1]=ye[i]
    coo_sorted_e=coo_e[np.lexsort(np.fliplr(coo_e).T)]
    for i in range(len(coo_sorted_e[:,0])):
        xes.append(coo_sorted_e[i,0])
        yes.append(coo_sorted_e[i,1])
    return xes,yes

#Compute pressure coefficient
def compute_cp(xes,yes,px,P1,name):
    ppx = Function(P1)
    ppx.interpolate(px)
    Cpe = []
    for i in range(len(xes)):
        Cpe.append(2*ppx.at(xes[i],yes[i]))
    file = open('data_SA_DA/Cp'+name+'.txt','w')
    for item in Cpe:
        file.write(str(item)+"\n")        

#Compute skin-friction coefficient    
def compute_cf(xes,yes,ux,P1,Re,name,write_true):
    #Extract component
    uu = Function(P1)
    uu.interpolate(ux[0])
    vv = Function(P1)
    vv.interpolate(ux[1])
    #Derivative
    dudx = Function(P1).interpolate(uu.dx(0))
    dudy = Function(P1).interpolate(uu.dx(1))    
    dvdx = Function(P1).interpolate(vv.dx(0))
    dvdy = Function(P1).interpolate(vv.dx(1))
    DUDXe=[]
    DUDYe=[]
    DVDXe=[]
    DVDYe=[]
    for i in range(len(xes)):
        DUDXe.append(dudx.at(xes[i],yes[i]))
        DUDYe.append(dudy.at(xes[i],yes[i]))
        DVDXe.append(dvdx.at(xes[i],yes[i]))
        DVDYe.append(dvdy.at(xes[i],yes[i]))
    Cfe=[]
    npoints=len(xes)
    for i in range(npoints):   
        if i==0: 
            t = (yes[1]-yes[npoints-1])/(xes[1]-xes[npoints-1])
        elif i==npoints-1: 
            t = (yes[0]-yes[npoints-2])/(xes[0]-xes[npoints-2])
        else: 
            t = (yes[i+1]-yes[i-1])/(xes[i+1]-xes[i-1])  
        nx=-t/sqrt(1. + t**2)
        ny=1/sqrt(1. + t**2)
        Cfe.append((2/Re)*(ny*(nx*2.*DUDXe[i] + ny*(DVDXe[i] + DUDYe[i])) 
				- nx*(ny*2.*DVDYe[i] + nx*(DUDYe[i] +DVDXe[i]))))  
    if write_true==True: 
        file = open('data_SA_DA/Cf'+name+'.txt','w')   
        for item in Cfe:
            file.write(str(item)+"\n")     
    return Cfe
    