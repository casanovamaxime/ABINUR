from utils.helpers_model_param import *

#Define variational residual
def define_var(ux,px,nutx,phiux,phipx,phinutx,nuNS,nuSA,PDC,tau_SUPG,tau_GRAD_DIV,dX):
    #NS equation
    N_NS =  nuNS*( 2.*(ux[0].dx(0))*(phiux[0].dx(0)) + 2.*(ux[1].dx(1))*(phiux[1].dx(1)) + ((ux[0].dx(1))+(ux[1].dx(0)))*(phiux[1].dx(0)) + ((ux[1].dx(0))+(ux[0].dx(1)))*(phiux[0].dx(1)) )*dX + (ux[0]*(ux[0].dx(0))+ux[1]*(ux[0].dx(1)))*phiux[0]*dX + (ux[0]*(ux[1].dx(0))+ux[1]*(ux[1].dx(1)))*phiux[1]*dX  - ((ux[0].dx(0)) + (ux[1].dx(1)))*phipx*dX - ((phiux[0].dx(0)) + (phiux[1].dx(1)))*px*dX 
    #NS stabilization
    N_NS_SUPG_GRAD_DIV =  tau_SUPG*(ux[0]*(phiux[0].dx(0))+ux[1]*(phiux[0].dx(1)))*(ux[0]*(ux[0].dx(0))+ux[1]*(ux[0].dx(1)))*dX + tau_SUPG*(ux[0]*(phiux[1].dx(0))+ux[1]*(phiux[1].dx(1)))*(ux[0]*(ux[1].dx(0))+ux[1]*(ux[1].dx(1)))*dX + tau_GRAD_DIV*((ux[0].dx(0))+(ux[1].dx(1)))*((phiux[0].dx(0))+(phiux[1].dx(1)))*dX
    #SA equation
    N_SA = (ux[0]*(nutx.dx(0)) + ux[1]*(nutx.dx(1)))*phinutx*dX - PDC*phinutx*dX + (1./sigma)*nuSA*((nutx.dx(0))*(phinutx.dx(0)) + (nutx.dx(1))*(phinutx.dx(1)))*dX 
    #SA stabilization
    N_SA_SUPG = tau_SUPG*(ux[0]*(phinutx.dx(0)) + ux[1]*(phinutx.dx(1)))*(ux[0]*(nutx.dx(0)) + ux[1]*(nutx.dx(1)))*dX
    #Total residual
    N=N_NS+N_NS_SUPG_GRAD_DIV+N_SA+N_SA_SUPG
    return N
