from utils.helpers_fe_bc_mesh import *

#SA constants
cv1 = 7.1
cb1 = 0.1355
cb2 = 0.622
sigma = 2./3
kk = 0.41
cw1 = cb1/(kk**2) + (1.+cb2)/sigma
cw2 = 0.3
cw3 = 2.
#NUM constants
M = 1e-4
MSUPG = 1e-1

#Define stabilization terms
def define_STAB(ux,h,nu):
    ht = 0.1*h
    NormeT = sqrt(ux[0]*ux[0] + ux[1]*ux[1] + MSUPG**2)
    Ret = 0.5*NormeT*ht/nu
    chiT = conditional(Ret<=3,Ret/3.,1.)
    tau_SUPG = chiT*ht/(2*NormeT)
    tau_GRAD_DIV = (ht*ht)/(8*tau_SUPG)
    return tau_SUPG,tau_GRAD_DIV

#Define diffusion terms
def define_DIFF(nutx,nu):
    xi = nutx/nu
    fv1 = (xi**3)/(abs(xi**3) + (cv1**3))
    fv2 = 1. - (xi/(1.+xi*fv1))
    nutt=nutx*fv1
    #in the SA equation
    nuSA = conditional(xi<0,nu*(1.+xi+0.5*(xi**2)),nu*(1.+xi))
    #in the NS equation
    nuNS = conditional(nutt>=0,nu+(1.)*nutt,nu)
    return xi,fv2,nuSA,nuNS

#Define production,destruction and cross-diffusion terms
def define_PDC(Qf,nutx,nu,d2x,xi,fv2,ux,fx,DA_TRUE):
    d2x=d2x+1e-12
    #Vorticity
    W = sqrt((ux[0].dx(1) - ux[1].dx(0))**2 + M) - M 
    Wb = nutx*fv2/(kk*kk*d2x)
    Wt = Wb + W
    rp =  nutx/(Wt*kk*kk*d2x)
    r = conditional(Or(rp<0,rp>10),10.,rp)
    g = r + cw2*(r**6 - r)
    fw = (1.)*g*((1.+(cw3**6))/((g**6) + (cw3**6)))**(1./6.)
    gn = 1. - (1000.*xi**2)/(1. + xi**2)
    grad_nutx = ((nutx.dx(0))**2 + (nutx.dx(1))**2)
    #Compute PDC
    if DA_TRUE==True:
        PDC = conditional(xi<0,(1.+(1./Qf)*fx)*cb1*W*nutx*gn + (cw1*nutx*nutx)/d2x +  (cb2/sigma)*grad_nutx, \
    (1.+(1./Qf)*fx)*cb1*Wt*nutx - (cw1*fw*nutx*nutx)/d2x + (cb2/sigma)*grad_nutx)
    else:
        PDC = conditional(xi<0,(1.+fx)*cb1*W*nutx*gn + (cw1*nutx*nutx)/d2x +  (cb2/sigma)*grad_nutx, \
    (1.+fx)*cb1*Wt*nutx - (cw1*fw*nutx*nutx)/d2x + (cb2/sigma)*grad_nutx)
    return PDC
