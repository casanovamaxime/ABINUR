from utils.helpers_model_solve import *

#Store cost functional value
j_s=[]
def eval_cb(j,a):
    j_s.append(j)
    print("J=",j)


#Perform data assimilation
def perform_DA(P1,ux,uvhfx_noise,fx,dX,tau,kappa2,f_opti_method,f_display,f_iter,f_tol,delta_y,sigma_y,DA_TRUE):
    #Compute fxbar
    phifx = TestFunction(P1)
    fxbar = Function(P1)
    solve(inner(fxbar, phifx)*dX - (tau)*(kappa2*inner(fx, phifx)*dX + inner(grad(fx), grad(phifx))*dX)==0, fxbar,bcs=[DirichletBC(P1, Constant(0.), [13,14])])
    #Where assimilating
    #Full cost functional
    if DA_TRUE==True:
        J = (1./2)*assemble( (fxbar**2)*dX )
        for i in range(len(delta_y)):
             J += (1./2)*assemble( (((ux[0]-uvhfx_noise[0])/sigma_y)**2)*(delta_y[i])*dX )
    #Observational cost functional
    else:
        J=0
        for i in range(len(delta_y)):
             J += (1./2)*assemble( (((ux[0]-uvhfx_noise[0])/sigma_y)**2)*(delta_y[i])*dX )
    #Control parameter
    m = Control(fx,riesz_map='l2')
    Jhat = ReducedFunctional(J, m,eval_cb_post=eval_cb)
    #Minimize automatically
    fAx = minimize(Jhat,options={"disp":f_display,"maxiter":f_iter,"gtol":f_tol,"ftol":f_tol})
    return fAx,Jhat

#Check gradient and hessian by finite difference
def check_grad_and_hess(Qf,Jhat,dire,Re_s,uvhfx,nuthfx,uvhfx_noise,qx,ux,px,nutx,phiux,phipx,phinutx,fx,P1,Q,h,d2x,dX,delta_y,sigma_y,annotate,method,DA_TRUE):
    #Compute gradient and hessian using Firedrake function
    grad_fire = Jhat.derivative()
    hess_action_d_fire = Jhat.hessian(dire,options={'riesz_representation': 'l2'})
    #Compute directional ones
    grad_fire_D = 0
    D_hess_D = 0
    for i in range(len(hess_action_d_fire.dat.data[:])):
        D_hess_D=D_hess_D + hess_action_d_fire.dat.data[i]*dire.dat.data[i]
        grad_fire_D=grad_fire_D + grad_fire.dat.data[i]*dire.dat.data[i]
    #Compute using finite difference
    epsi=1e-4#1e-3 work
    #First forward solve
    w = solve_SA_BorA(uvhfx,nuthfx,Qf,Re_s,qx,ux,px,nutx,phiux,phipx,phinutx,fx,Q,h,d2x,dX,annotate,method,DA_TRUE)
    Ju=0
    for i in range(len(delta_y)):
        Ju += assemble(0.5*((1./(sigma_y**2))*((ux[0]-uvhfx_noise[0])**2)*(delta_y[i]))*dx)
    #Second forward solve + epsi
    fx_plus = Function(P1)
    for it in range(len(fx.dat.data[:])):
        fx_plus.dat.data[it]=fx.dat.data[it]+epsi*dire.dat.data[it]
    w = solve_SA_BorA(uvhfx,nuthfx,Qf,Re_s,qx,ux,px,nutx,phiux,phipx,phinutx,fx_plus,Q,h,d2x,dX,annotate,method,DA_TRUE)
    Jplusu=0
    for i in range(len(delta_y)):
        Jplusu += assemble(0.5*((1./(sigma_y**2))*((ux[0]-uvhfx_noise[0])**2)*(delta_y[i]))*dx)
    #Third forward solve - epsi
    fx_minus = Function(P1)
    for it in range(len(fx.dat.data[:])):
        fx_minus.dat.data[it]=fx.dat.data[it]-epsi*dire.dat.data[it]
    w = solve_SA_BorA(uvhfx,nuthfx,Qf,Re_s,qx,ux,px,nutx,phiux,phipx,phinutx,fx_minus,Q,h,d2x,dX,annotate,method,DA_TRUE)
    Jminusu=0
    for i in range(len(delta_y)):
        Jminusu += assemble(0.5*((1./(sigma_y**2))*((ux[0]-uvhfx_noise[0])**2)*(delta_y[i]))*dx)
    #Compare
    D_HESS_D_df = (Jplusu+Jminusu-2*Ju)/(epsi*epsi)
    GRAD_D_df = (Jplusu-Jminusu)/(2*epsi)
    print("COMPA HESS : FIRE =",D_hess_D,"  |  DF = ",D_HESS_D_df)
    print("COMPA GRAD : FIRE =",grad_fire_D,"  |  DF = ",GRAD_D_df)
    return D_HESS_D_df,D_hess_D
