from utils.helpers_model_var import *


#Solve nonlinear problem using Newton
def solve_newton(N,qx,bcs,nullspace,annotate):
    solve(N == 0, qx, bcs=bcs, nullspace=nullspace,
                              solver_parameters={"snes_monitor": None, "snes_type": "newtonls","snes_linesearch_damping":1, "snes_max_it" : 5000,
                              "ksp_type": "none",
                              "mat_type": "aij",
                              "pc_type": "lu",
                              "pc_factor_mat_solver_type": "mumps"}, 
                               annotate=annotate)
    return qx


#Loop over Reynolds number and solve
def solve_SA_BorA(uvhfx,nuthfx,Qf,Re_s,qx,ux,px,nutx,phiux,phipx,phinutx,fx,Q,h,d2x,dX,annotate,method,DA_TRUE):
    for i in range(len(Re_s)):
        Re = Re_s[i]
        nu = 1./Re
        #Define stabilization terms
        tau_SUPG,tau_GRAD_DIV = define_STAB(ux,h,nu)
        #Define diffusion terms
        xi,fv2,nuSA,nuNS = define_DIFF(nutx,nu)
        #define production, destruction and cross-diffusion terms
        PDC = define_PDC(Qf,nutx,nu,d2x,xi,fv2,ux,fx,DA_TRUE)
        #Define variational residual
        N = define_var(ux,px,nutx,phiux,phipx,phinutx,nuNS,nuSA,PDC,tau_SUPG,tau_GRAD_DIV,dX)
        #Define boundary conditions
        bcs,nullspace = define_bc(Q,nu,uvhfx,nuthfx)
        #Solve variational problem
        qx=solve_newton(N,qx,bcs,nullspace,annotate)
        print(i+1,"/",len(Re_s),": Re = ",Re_s[i]," OK")
    return qx
