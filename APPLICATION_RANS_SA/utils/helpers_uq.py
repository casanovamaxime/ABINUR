from utils.helpers_da import *
from matplotlib.colors import LogNorm
from petsc4py import PETSc as pet
from mpi4py.MPI import COMM_WORLD as comm
from petsc4py import PETSc
from slepc4py import SLEPc 

#A prior UQ 
def perform_UQ_B(xes,yes,sigma_y,uvhfx,nuthfx,uvhfx_noise,Mf,Qf,Kf,Re_s,qx,ux,px,nutx,phiux,phipx,phinutx,fx,P1,P1B,Q,h,d2x,dX,rB,number_of_dir_to_check,check_grad_hess,annotate,method,DA_TRUE):
    #Read last Reynolds
    nu = (1./Re_s[-1])
    #Define stabilization terms
    tau_SUPG,tau_GRAD_DIV = define_STAB(ux,h,nu)
    #Define diffusion terms
    xi,fv2,nuSA,nuNS = define_DIFF(nutx,nu)
    #Define variational residual
    PDC = define_PDC(Qf,nutx,nu,d2x,xi,fv2,ux,fx,DA_TRUE)
    #Define boundary conditions
    N = define_var(ux,px,nutx,phiux,phipx,phinutx,nuNS,nuSA,PDC,tau_SUPG,tau_GRAD_DIV,dX)
    #Solve variational problem
    bcs,nullspace = define_bc(Q,nu,uvhfx,nuthfx)
    perform_EIGEN_B(Re_s,xes,yes,sigma_y,Mf,Qf,Kf, N, qx,ux,px,nutx, fx, bcs, rB, P1, P1B, Q, method, dX)
    
#A posteriori UQ 
def perform_UQ_A(xes,yes,delta_y,sigma_y,uvhfx,nuthfx,uvhfx_noise,Mf,Qf,Kf,Jhat,Re_s,qx,ux,px,nutx,phiux,phipx,phinutx,fx,P1,P1B,Q,h,d2x,dX,rA,number_of_dir_to_check,check_grad_hess,annotate,method,DA_TRUE):
    #Read last Reynolds
    nu = (1./Re_s[-1])
    #Define stabilization terms
    tau_SUPG,tau_GRAD_DIV = define_STAB(ux,h,nu)
    #Define diffusion terms
    xi,fv2,nuSA,nuNS = define_DIFF(nutx,nu)
    #Define variational residual
    PDC = define_PDC(Qf,nutx,nu,d2x,xi,fv2,ux,fx,DA_TRUE)
    #Define boundary conditions
    N = define_var(ux,px,nutx,phiux,phipx,phinutx,nuNS,nuSA,PDC,tau_SUPG,tau_GRAD_DIV,dX)
    #Solve variational problem
    bcs,nullspace = define_bc(Q,nu,uvhfx,nuthfx)
    #Perforrm eigendecomposition of the posterior matrix
    all_vecs = perform_EIGEN_A(Re_s,xes,yes,delta_y,sigma_y,Mf,Qf,Kf, N, qx,ux,px,nutx, fx, Jhat, bcs, rA, P1, P1B, Q, method, dX)
    if check_grad_hess == True:
        df_hessl=[]
        fire_hessl=[]
        for i in range(number_of_dir_to_check):
            dire = Function(P1)
            #print(all_vecs[i][:])
            dire.dat.data[:] = all_vecs[i][:]
            method='A'
            df_hess,fire_hess = check_grad_and_hess(Qf,Jhat,dire,Re_s,uvhfx,nuthfx,uvhfx_noise,qx,ux,px,nutx,phiux,phipx,phinutx,fx,P1,Q,h,d2x,dX,delta_y,sigma_y,annotate,method,DA_TRUE) 
            df_hessl.append(df_hess)
            fire_hessl.append(fire_hess)
        file = open('data_UQA/df_hess.txt','w')
        for item in df_hessl:
            file.write(str(item)+"\n")   
        file = open('data_UQA/fire_hess.txt','w')
        for item in fire_hessl:
            file.write(str(item)+"\n")  

#For matrix inversion              
class MATinv(object):
    def __init__(self, MAT):  
        self.MAT = MAT
                                              
    def assemble_MATinv(self):
        self.MATinv = self.MAT
        self.MATinv_hT = self.MATinv.copy().hermitianTranspose()
                                        
    def linear_problem(self, A, b, x):
        ksp = pet.KSP()
        ksp.create(comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        PC = ksp.getPC()
        PC.setType("lu")
        PC.setFactorSolverType("mumps")
        ksp.solve(b, x)
        x.ghostUpdate(addv=pet.InsertMode.INSERT,
                      mode=pet.ScatterMode.FORWARD)
                      
#======================================================= EVP PRIOR B ========================================================
def perform_EIGEN_B(Re_s,xes,yes,sigma_y,Mf,Qf,Kf, N, qx,ux,px,nutx, fx, bcs, rB, P1, P1B, Q, method, dX):
    #Eigenforcing and eigenstates
    fBxk = Function(P1)
    uBxk = Function(P1B)
    pBxk = Function(P1)
    eBxk = Function(P1)
    #Variance
    sigma2_fBx = Function(P1)
    sigma2_uBx = Function(P1B)
    sigma2_pBx = Function(P1)   
    sigma2_eBx = Function(P1)
    sigma2_CfB = []
    npoints=len(xes)
    for i in range(npoints):  
        sigma2_CfB.append(0.)
    #Dimension of matrices
    ndof = int(len(fBxk.dat.data[:]))
    ndofq = int(2*len(uBxk.dat.data[:]) + 2*len(fBxk.dat.data[:]))
    ndofu = int(2*len(uBxk.dat.data[:]))
    ndofp = int(len(pBxk.dat.data[:]))
    #Temporar vectors for matrix-vector computations in slepc
    temp1 = pet.Vec().createSeq(ndof)
    temp2 = pet.Vec().createSeq(ndof)
    temp3 = pet.Vec().createSeq(ndofq)
    temp4 = pet.Vec().createSeq(ndofq)

    #dNdf matrix               
    dNdf = assemble(derivative(N,fx)).M.handle
    #dNdq matrix        
    dNdq = assemble(derivative(N,qx),bcs=bcs).M.handle
    #(dNdu)^{-1}
    dNdq_inv = MATinv(dNdq)
    dNdq_inv.assemble_MATinv()
    #Kf matrix
    Kf = Kf.M.handle
    #(Kf)^{-1}
    Kf_inv = MATinv(Kf)
    Kf_inv.assemble_MATinv() 
    #Mf mass matrix full
    Mf = Mf.M.handle
    n = ndof    
    #Operator class for eigendecomposition
    #Prior covariance matrix
    class Operator_BFB :
        def __init__(self, Kf_inv,Qf):
            self.Kf_inv = Kf_inv  # PETSc Mat from assemble(a).M.handle
            self.Qf = Qf
        def mult(self, mat, x, y):
            temp2[:]=x[:]*self.Qf.dat.data[:]
            self.Kf_inv.linear_problem(self.Kf_inv.MATinv, temp2, temp1) 
            temp2[:]= temp1[:]*self.Qf.dat.data[:]**2
            self.Kf_inv.linear_problem(self.Kf_inv.MATinv, temp2, temp1) 
            y[:]=temp1[:]*self.Qf.dat.data[:]   
   # Assemble PETSc matrices from Firedrake
    A = PETSc.Mat().createPython([n, n])
    A.setPythonContext(Operator_BFB(Kf_inv,Qf))
    A.setUp()
    # Create eigensolver
    E = SLEPc.EPS().create()
    E.setOperators(A)
    E.setProblemType(SLEPc.EPS.ProblemType.HEP)  
    E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    # Solver settings
    E.setDimensions(nev=rB)     
    st = E.getST()
    ksp = st.getKSP()
    ksp.setType('gmres')
    ksp.getPC().setType('none')
    E.setFromOptions()
    #Solve the problem
    t_start = time.time() 
    E.solve()
    t_end = time.time() 
    print("TIME SLEPC:",t_end-t_start)
    
    #Number of total converged lambda_k
    nconv = E.getConverged()
    print(f"Converged eigenpairs: {nconv}")
    lambda_k=[]
    for k in range(0,nconv):
        eig = E.getEigenvalue(k).real   
        lambda_k.append(eig)
        vr, vi = A.getVecs() 
        E.getEigenvector(k, vr, vi)
        fxk = vr.getArray()   
        #Check scalar prodcut 
        print("\lambda_B"+str(k),lambda_k[k])
        print("<fBx"+str(k)+",fBx"+str(k)+">=",np.sum((fxk[:])**2))  
                                    
        #Eigenforcing
        fBxk.dat.data[:] = fxk[:]/Qf.dat.data[:] 
        temp2[:] = fBxk.dat.data[:]       
        dNdf.mult(temp2, temp3)
        dNdq_inv.linear_problem(dNdq_inv.MATinv, temp3, temp4)
        #Eigenvelocity
        uBxk.dat.data[:] = np.array([-temp4.array[0:int(ndofu/2)],-temp4.array[int(ndofu/2):2*int(ndofu/2)]]).reshape((int(ndofu/2),2))
        #Eigenpressure
        pBxk.dat.data[:] = -temp4[2*int(ndofu/2):2*int(ndofu/2)+ndofp]
        eBxk.dat.data[:] = -temp4[2*int(ndofu/2)+ndofp:]
        
        #prior variance f
        sigma2_fBx.dat.data[:] += ( (lambda_k[k])*( fBxk.dat.data[:])**2)
        #prior variance u
        sigma2_uBx.dat.data[:,0] += ( (lambda_k[k])*( uBxk.dat.data[:,0])**2)
        sigma2_uBx.dat.data[:,1] += ( (lambda_k[k])*( uBxk.dat.data[:,1])**2)
        #prior variance u
        sigma2_pBx.dat.data[:] += ( (lambda_k[k])*( pBxk.dat.data[:])**2)
        sigma2_eBx.dat.data[:] += ( (lambda_k[k])*( eBxk.dat.data[:])**2)
        CfA = compute_var_cfA(xes,yes,uBxk,P1,Re_s[-1])
        for pp in range(len(xes)):
            sigma2_CfB[pp]+=( lambda_k[k] )*(CfA[pp]**2)
        
    #prior standard deviation 
    sigma2_fBx.dat.data[:] = np.sqrt(sigma2_fBx.dat.data[:])
    sigma2_uBx.dat.data[:,0] = np.sqrt(sigma2_uBx.dat.data[:,0])
    sigma2_uBx.dat.data[:,1] = np.sqrt(sigma2_uBx.dat.data[:,1])
    sigma2_pBx.dat.data[:] = np.sqrt(sigma2_pBx.dat.data[:])
    sigma2_eBx.dat.data[:] = np.sqrt(sigma2_eBx.dat.data[:])
     
    file = open('data_UQB/lambda'+method+'_.txt','w')
    for item in lambda_k:
        file.write(str(item)+"\n")
    file = open('data_UQB/sigma_f'+method+'_.txt','w')    
    for item in sigma2_fBx.dat.data[:]:
        file.write(str(item)+"\n")
    file = open('data_UQB/sigma2_CfB'+method+'_.txt','w')    
    for item in sigma2_CfB[:]:
        file.write(str(item)+"\n")
    file = open('data_UQB/sigma_u'+method+'_.txt','w')    
    for item in sigma2_uBx.dat.data[:,0]:
        file.write(str(item)+"\n")
    file = open('data_UQB/sigma_v'+method+'_.txt','w')    
    for item in sigma2_uBx.dat.data[:,1]:
        file.write(str(item)+"\n")      
    file = open('data_UQB/sigma_p'+method+'_.txt','w')    
    for item in sigma2_pBx.dat.data[:]:
        file.write(str(item)+"\n")
    file = open('data_UQB/sigma_e'+method+'_.txt','w')    
    for item in sigma2_eBx.dat.data[:]:
        file.write(str(item)+"\n")    
    
#======================================================= EVP POSTERIOR A ========================================================
def perform_EIGEN_A(Re_s,xes,yes,delta_y,sigma_y,Mf,Qf,Kf, N, qx,ux,px,nutx, fx, Jhat, bcs, rA, P1, P1B, Q, method, dX):   
    #Eigenforcing and eigenstates
    fAxk = Function(P1)
    uAxk = Function(P1B)
    pAxk = Function(P1)
    eAxk = Function(P1)
    #Variance
    sigma2_fAx = Function(P1)
    sigma2_uAx = Function(P1B)
    sigma2_pAx = Function(P1)   
    sigma2_eAx = Function(P1)
    sigma2_CfA = []
    npoints=len(xes)
    for i in range(npoints):  
        sigma2_CfA.append(0.)
    #Dimension of matrices
    ndof = int(len(fAxk.dat.data[:]))
    ndofq = int(2*len(uAxk.dat.data[:]) + 2*len(fAxk.dat.data[:]))
    ndofu = int(2*len(uAxk.dat.data[:]))
    ndofp = int(len(pAxk.dat.data[:]))
    #Temporar vectors for matrix-vector computations in slepc
    temp1 = pet.Vec().createSeq(ndof)
    temp2 = pet.Vec().createSeq(ndof)
    temp3 = pet.Vec().createSeq(ndofq)
    temp4 = pet.Vec().createSeq(ndofq)
    temp5 = pet.Vec().createSeq(int(ndofu/2))
    temp6 = pet.Vec().createSeq(int(ndofu/2))

    #dNdf matrix               
    dNdf = assemble(derivative(N,fx)).M.handle
    #dNdq matrix        
    dNdq = assemble(derivative(N,qx),bcs=bcs).M.handle
    #(dNdu)^{-1}
    dNdq_inv = MATinv(dNdq)
    dNdq_inv.assemble_MATinv()
    #Kf matrix
    Kf = Kf.M.handle
    #(Kf)^{-1}
    Kf_inv = MATinv(Kf)
    Kf_inv.assemble_MATinv() 
    #Mf mass matrix full
    Mf = Mf.M.handle
    #Observation
    ny = len(delta_y)
    print("ny=",ny)
    #scalar product
    trial1,trial2,trial3 = split(TrialFunction(Q))
    test1,test2,test3 = split(TestFunction(Q))
    my=0
    for i in range(ny):
        my+=(1./(sigma_y**2))*delta_y[i]*inner(trial1[0],test1[0])
    My = assemble(my*dx).M.handle
    n = ndof   
    #Operator class for eigendecomposition
    #Linear prior-preconditioned observational hessian 
    class Operator_HFA_GN:
        def __init__(self,dNdq_inv,dNdf,My,Kf_inv,Qf):
            self.dNdq_inv = dNdq_inv
            self.dNdf = dNdf
            self.My = My
            self.Kf_inv = Kf_inv
            self.Qf = Qf
        def mult(self, mat, x, y): 
            temp2[:]=x[:]*self.Qf.dat.data[:]
            self.Kf_inv.linear_problem(self.Kf_inv.MATinv, temp2, temp1) 
            self.dNdf.mult(temp1,temp3)
            self.dNdq_inv.linear_problem(self.dNdq_inv.MATinv,temp3,temp4)
            self.My.mult(temp4,temp3) #scalar product in the observation domain
            self.dNdq_inv.linear_problem(self.dNdq_inv.MATinv_hT,temp3,temp4)
            self.dNdf.multTranspose(temp4,temp1)
            self.Kf_inv.linear_problem(self.Kf_inv.MATinv, temp1, temp2) 
            y[:]=temp2[:]*self.Qf.dat.data[:]
    #Full prior-preconditioned observational hessian 
    class Operator_HFA:
        def __init__(self,Jhat,Kf_inv,Qf):
            self.Jhat = Jhat
            self.Kf_inv = Kf_inv
            self.Qf = Qf
        def mult(self, mat, x, y): 
            temp2[:]=x[:]*self.Qf.dat.data[:]
            self.Kf_inv.linear_problem(self.Kf_inv.MATinv, temp2, temp1) 
            fAxk.dat.data[:]=temp1[:]
            temp2[:]= self.Jhat.hessian(fAxk,options={'riesz_representation': 'l2'}).dat.data[:]
            self.Kf_inv.linear_problem(self.Kf_inv.MATinv, temp2, temp1) 
            y[:]=temp1[:]*self.Qf.dat.data[:]      
   #Assemble PETSc matrices from Firedrake
    A = PETSc.Mat().createPython([n, n])
    #A.setPythonContext(Operator_HFA(Jhat,Kf_inv,Qf))
    A.setPythonContext(Operator_HFA_GN(dNdq_inv,dNdf,My,Kf_inv,Qf))
    A.setUp()
    #Create eigensolver
    E = SLEPc.EPS().create()
    E.setOperators(A)
    E.setProblemType(SLEPc.EPS.ProblemType.HEP)  # Hermitian generalized problem
    E.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    #Solver settings
    E.setDimensions(nev=rA)#Number of lambda_k to compute
    st = E.getST()
    ksp = st.getKSP()
    ksp.setType('gmres')
    ksp.getPC().setType('none')
    E.setFromOptions()
    #Solve the problem
    t_start = time.time() 
    E.solve()
    t_end = time.time() 
    print("TIME SLEPC:",t_end-t_start)
    #Number of total converged lambda_k
    nconv = E.getConverged()
    print(f"Converged eigenpairs: {nconv}")
    lambda_k=[]
    lambda_u_k=[]
    lambda_f_k=[]
    check_eigenvectors=[]
    for k in range(0,nconv):
        eig = E.getEigenvalue(k).real   
        lambda_k.append(eig)
        vr, vi = A.getVecs() 
        E.getEigenvector(k, vr, vi)
        fxk = vr.getArray()   
        #Check scalar prodcut 
        print("\lambda_A"+str(k),lambda_k[k])
        print("<fAx"+str(k)+",fAx"+str(k)+">=",np.sum(fxk[:]**2)) 
           
        #For check eigenvalue with FD
        temp1[:] = fxk.real[:]*Qf.dat.data[:]
        Kf_inv.linear_problem(Kf_inv.MATinv, temp1, temp2) 
        check_eigenvectors.append(temp2[:])
        
        #Compute eigendirections     
        temp1[:] = fxk.real[:]*Qf.dat.data[:]
        Kf_inv.linear_problem(Kf_inv.MATinv, temp1, temp2) 
        #Eigenforcing
        fAxk.dat.data[:]=temp2[:]     
        lambda_f_k.append((lambda_k[k]/(1.+lambda_k[k]))*assemble(inner(fAxk,fAxk)*dx))  
        dNdf.mult(temp2, temp3)
        dNdq_inv.linear_problem(dNdq_inv.MATinv, temp3, temp4)
        #Eigenvelocity
        uAxk.dat.data[:] = np.array([-temp4.array[0:int(ndofu/2)],-temp4.array[int(ndofu/2):2*int(ndofu/2)]]).reshape((int(ndofu/2),2))
        lambda_u_k.append((lambda_k[k]/(1.+lambda_k[k]))*assemble(inner(uAxk[0],uAxk[0])*dx))  
        #Eigenpressure
        pAxk.dat.data[:] = -temp4[2*int(ndofu/2):2*int(ndofu/2)+ndofp]
        eAxk.dat.data[:] = -temp4[2*int(ndofu/2)+ndofp:]
                
        #posterior preconditioned variance f (need to remove this from the prior)
        sigma2_fAx.dat.data[:] += ( lambda_k[k]/(1.+lambda_k[k]) )*(fAxk.dat.data[:])**2   
        #posterior preconditioned variance u (need to remove this from the prior)      
        sigma2_uAx.dat.data[:,0] += ( lambda_k[k]/(1.+lambda_k[k]) )*(uAxk.dat.data[:,0])**2 
        sigma2_uAx.dat.data[:,1] += ( lambda_k[k]/(1.+lambda_k[k]) )*(uAxk.dat.data[:,1])**2
        #posterior preconditioned variance p (need to remove this from the prior)
        sigma2_pAx.dat.data[:] += ( lambda_k[k]/(1.+lambda_k[k]) )*(pAxk.dat.data[:])**2 
        sigma2_eAx.dat.data[:] += ( lambda_k[k]/(1.+lambda_k[k]) )*(eAxk.dat.data[:])**2 
        #posterior variance Cf (need to remove this from the prior)
        CfA = compute_var_cfA(xes,yes,uAxk,P1,Re_s[-1])
        for pp in range(len(xes)):
            sigma2_CfA[pp]+=( lambda_k[k]/(1.+lambda_k[k]) )*(CfA[pp]**2)
        
    #posterior preconditioned standard deviation (need to remove this from the prior)
    sigma2_fAx.dat.data[:] = np.sqrt(sigma2_fAx.dat.data[:])
    sigma2_uAx.dat.data[:,0] = np.sqrt(sigma2_uAx.dat.data[:,0])
    sigma2_uAx.dat.data[:,1] = np.sqrt(sigma2_uAx.dat.data[:,1])
    sigma2_pAx.dat.data[:] = np.sqrt(sigma2_pAx.dat.data[:])
    sigma2_eAx.dat.data[:] = np.sqrt(sigma2_eAx.dat.data[:])
    
    file = open('data_UQA/lambdaA_fa_.txt','w')
    for item in lambda_f_k:
        file.write(str(item)+"\n")
    file = open('data_UQA/lambdaA_ua_.txt','w')
    for item in lambda_u_k:
        file.write(str(item)+"\n")
    file = open('data_UQA/lambdaA_.txt','w')
    for item in lambda_k:
        file.write(str(item)+"\n")
    file = open('data_UQA/sigma2_CfA.txt','w')    
    for item in sigma2_CfA[:]:
        file.write(str(item)+"\n")
    file = open('data_UQA/sigma_fA_.txt','w')    
    for item in sigma2_fAx.dat.data[:]:
        file.write(str(item)+"\n")
    file = open('data_UQA/sigma_uA_.txt','w')    
    for item in sigma2_uAx.dat.data[:,0]:
        file.write(str(item)+"\n")
    file = open('data_UQA/sigma_vA_.txt','w')    
    for item in sigma2_uAx.dat.data[:,1]:
        file.write(str(item)+"\n")      
    file = open('data_UQA/sigma_pA_.txt','w')    
    for item in sigma2_pAx.dat.data[:]:
        file.write(str(item)+"\n")
    file = open('data_UQA/sigma_eA_.txt','w')    
    for item in sigma2_eAx.dat.data[:]:
        file.write(str(item)+"\n")    
    
    return check_eigenvectors
    

def compute_var_cfA(xes,yes,ux,P1,Re):
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
    npoints=len(xes)
    CfA=[]
    for i in range(npoints):   
        if i==0: 
            t = (yes[1]-yes[npoints-1])/(xes[1]-xes[npoints-1])
        elif i==npoints-1: 
            t = (yes[0]-yes[npoints-2])/(xes[0]-xes[npoints-2])
        else: 
            t = (yes[i+1]-yes[i-1])/(xes[i+1]-xes[i-1])  
        nx=-t/sqrt(1. + t**2)
        ny=1/sqrt(1. + t**2)
        CfA.append( ((2/Re)*(ny*(nx*2.*DUDXe[i] + ny*(DVDXe[i] + DUDYe[i])) 
				- nx*(ny*2.*DVDYe[i] + nx*(DUDYe[i] +DVDXe[i])))) )    
    return CfA
