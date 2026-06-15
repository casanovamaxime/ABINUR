import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import time
import meshio as meshio
from firedrake import *
from firedrake.adjoint import *
from firedrake.__future__ import interpolate
import numpy as np
import matplotlib.tri as tri
from firedrake.pyplot import tripcolor, tricontour,tricontourf,triplot
from matplotlib.tri import Triangulation
import firedrake.pyplot as fplt
from matplotlib.colors import Normalize
from matplotlib.colors import BoundaryNorm
parameters["reorder_meshes"] = False
parameters["reorder_dofs_serial"] = False
    
#Define CDC mesh
def define_mesh():
    mesh = Mesh('mesh/cdc.msh')
    h = CellDiameter(mesh)
    n = FacetNormal(mesh)
    return mesh,h,n

#Define function spaces 
def define_fe_space(mesh):
    #Define function spaces
    P0 = FunctionSpace(mesh,"DG", 0)
    P1 = FunctionSpace(mesh,"CG", 1)
    P1B = FunctionSpace(mesh, VectorElement(NodalEnrichedElement(FiniteElement('P',triangle,1), FiniteElement('B',triangle,3))))
    #Mixed function space
    Q = P1B * P1 * P1
    return P0,P1,P1B,Q

#Define fucntions and test functions
def define_qx_phiqx(Q):
    #State vector function
    qx = Function(Q)
    ux,px,nutx = split(qx)
    #Basis functions
    phiux,phipx,phinutx = TestFunctions(Q)
    return qx,ux,px,nutx,phiux,phipx,phinutx

#Define boundary conditions
def define_bc(Q,nu,uvhfx,nuthfx):
    #BC bottom wall
    bcu_bottom = DirichletBC(Q.sub(0), Constant((0, 0)), [11])
    bcnu_bottom = DirichletBC(Q.sub(2), Constant(0), [11])
    #BC top wall
    bcu_top = DirichletBC(Q.sub(0), Constant((0, 0)), [12])
    bcnu_top = DirichletBC(Q.sub(2), Constant(0), [12])
    #BC inlet        
    bcu_in = DirichletBC(Q.sub(0), uvhfx, [13])
    bcnu_in = DirichletBC(Q.sub(2), nuthfx, [13])
    #FULL bc  
    bcs = [bcu_bottom,bcnu_bottom,bcu_in,bcnu_in,bcu_top,bcnu_top]
    nullspace = MixedVectorSpaceBasis(Q, [Q.sub(0), VectorSpaceBasis(constant=True), Q.sub(2)])
    return bcs,nullspace
    
#compute squared distance to the wall
def distance2(P0,P1,Md):
    #Linear solution
    phiddlx = TestFunction(P1)
    ddlx = TrialFunction(P1)
    #Nonlinear solution
    ddx = Function(P1)
    #Initialization problem to get good initial guess for nonlinear problem:
    F1 = inner(grad(ddlx), grad(phiddlx))*dx - Constant(1.0)*phiddlx*dx
    solve(lhs(F1)==rhs(F1), ddx, DirichletBC(P1, 0, [11,12]))
    #Stabilized Eikonal equation
    F = sqrt(inner(grad(ddx), grad(ddx)))*phiddlx*dx - Constant(1.0)*phiddlx*dx + Md*inner(grad(ddx), grad(phiddlx))*dx
    solve(F==0, ddx, DirichletBC(P1, 0, [11,12]))
    #Compute squared distance
    d2x = Function(P0)
    phid2x = TestFunction(P0)
    F = d2x*phid2x*dx - (ddx*ddx)*phid2x*dx
    solve(F==0,d2x)
    return d2x