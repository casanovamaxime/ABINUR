from utils.helpers_uq import *

#Read high fidelity observations
def read_field_HF(P1,P1B):
    #Read nuthf
    meshnut = meshio.read('data_HF/nut_DNS.vtk')
    nutbl_loc=meshnut.point_data["nute"]
    nutbl=Function(P1)
    nutbl.dat.data[:] = nutbl_loc[:]
    phitnuthfx = TestFunction(P1)
    nuthfx = Function(P1)
    F = inner(nuthfx,phitnuthfx)*dx - nutbl*phitnuthfx*dx
    solve(F==0, nuthfx)     
    #Read uhf
    meshu = meshio.read('data_HF/nut_DNS.vtk')
    ules_loc=meshu.point_data["ue"]
    uhfx=Function(P1)
    uhfx.dat.data[:] = ules_loc[:]
    #Read vhf
    meshv = meshio.read('data_HF/nut_DNS.vtk')
    vles_loc=meshv.point_data["ve"]
    vhfx=Function(P1)
    vhfx.dat.data[:] = vles_loc[:]
    #Buil uvhf
    phiuhfx = TestFunction(P1B)
    uvhfx = Function(P1B)
    F = inner(uvhfx,phiuhfx)*dx - uhfx*phiuhfx[0]*dx- vhfx*phiuhfx[1]*dx 
    solve(F==0, uvhfx)
    return uvhfx,nuthfx

#Add noise in the observations
def define_noisy_HF(etax,P1,P1B):#dnsux
    uvhfx2,nuthfx2 = read_field_HF(P1,P1B)
    uvhfx2.dat.data[:,0]+=etax.dat.data[:,0]
    uvhfx_noise = Function(P1B)
    phiuvhfx_noise = TestFunction(P1B)
    F = inner(uvhfx_noise,phiuvhfx_noise)*dx - inner(uvhfx2,phiuvhfx_noise)*dx
    #F = inner(uvhfx_noise,phiuvhfx_noise)*dx - inner(dnsux + 0.000000001*etax,phiuvhfx_noise)*dx
    solve(F==0,uvhfx_noise)
    return uvhfx_noise

#Read Cp Cf LES
def read_cp_cf_LES():
    cples=[]
    cfles=[]
    xcfles=[]
    xcples=[]
    with open('data_HF/wall.txt') as f:
        lines = f.readlines()
    for it,el in enumerate(lines):
        if it>=29 and it <=2300:
            data=el.split()
            #print(it,data)
            xcfles.append(float(data[0]))
            cfles.append(float(data[6]))
            
    with open('data_HF/wall.txt') as f:
        lines = f.readlines()
    for it,el in enumerate(lines):
        if it>=29 and it <=2300:
            data=el.split()
            xcples.append(float(data[0]))
            cples.append(-float(data[4]))
    return xcfles,xcples,cples,cfles
    
#Compute approximated dirac function to peform point data assimilation
def delta_y(x,y,y_points,P1,sigma):
    delta_y = []
    for (cx, cy) in y_points:
        dist2 = (x - float(cx))**2 + (y - float(cy))**2
        delta_y.append( Function(P1).interpolate((1./((sigma**2)*(2*pi)))*exp( -0.5*(1./(sigma**2))*(dist2) ) ) )
    return delta_y
