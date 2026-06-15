from utils.helpers_uq import *

#Define approximation of the mass matrix by lumping
def define_Qf(P1):
    mass_form = inner(TestFunction(P1),TrialFunction(P1))*dx
    c1 = Function(P1).interpolate(Constant((1)))
    mass_action_form_a_sqrt = np.sqrt(assemble(action(mass_form,c1)).dat.data)
    #Cholesky factorization of the lumped mass matrix M = Qf^T Qf
    Qf = Function(P1)
    Qf.dat.data[:]=mass_action_form_a_sqrt[:]
    return Qf

#Define matrix K 
def define_Kf(P1,kappa2_f,tau_f):
    k = (tau_f)*( kappa2_f*inner(TrialFunction(P1), TestFunction(P1))*dx + inner(grad(TrialFunction(P1)), grad(TestFunction(P1)))*dx )
    #Apply Dirichlet bc on farfield (no variance expected)
    Kf = assemble(k,bcs=[DirichletBC(P1, Constant(0.), [13,14])])
    return Kf

#Define full mass matrix
def define_Mf(P1):
    m = inner(TrialFunction(P1),TestFunction(P1))*dx  
    #Full mass matrix
    Mf = assemble(m)
    return Mf
                    
