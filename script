import datetime
from fileinput import filename
# using now() to get current time  
time_start = datetime.datetime.now()
print(time_start,"starting ...")  
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"             # had a warning that it might run best on 1 thread
from firedrake import *

import time
import sys
import traceback

### code source code file ###
from shutil import copy
from ufl.tensors import as_scalar
from ufl.operators import div, grad, nabla_grad
from ufl import indexed
from cmath import sqrt
from firedrake.utility_meshes import PeriodicIntervalMesh
from ufl.differentiation import NablaGrad


def err(type, value, tb):
	print("\n")
	print(datetime.datetime.now(),"Exception after",datetime.datetime.now()-time_start)
	sys.__excepthook__(type, value, tb)
	print("\n")

sys.excepthook = err


print(datetime.datetime.now(),"import complete")
simulationId = str(time_start) +str("_-_") +str(int(np.round(10000000000*np.random.rand())))
simulationId = simulationId.replace(":","-")
simulationId = simulationId.replace(" ","_")
infoString = "simulation info"
infoString += "\n\t"+"simulationId"+" = \t\t"+str(simulationId)
infoString += "\n\t"+"time_start"+" = \t\t"+str(time_start)





### PARAMETERS ###
n = 200
tau = 0.000005

Lx = 1
Ly = 1
tEnd = 0.01
alpha = Constant(0)			# alpha = 1/L_s in navier slip
nu = Constant(1.0)			# ... - nu * Laplace u ...
kappa = Constant(1.0)			# ... - kappa * Laplace theta ...
Ra = Constant(10**8)			# ... + Ra * theta * e_2
Pr = Constant(1.0)			# 1/Pr*(u_t+u cdot nabla u) + ...

problem = "rayleighBenard"		# either 
					#	boussinesq (no boundary for theta only ic) 
					#	rayleighBenard (boundary -1, 1, ic = 0 + match boundary)

### PARAMETERS END ###


### OPTIONS ###

projectPoutputToAverageFree = True 			# force the output function of p to be average free (doesn't change the calculation)

writeOutputEvery = 0.000025

outputFolder = "output/"
dataFolder = outputFolder + "data/"



nx = round(n*Lx/Ly)
ny = n

### OPTIONS END ###

mesh = PeriodicRectangleMesh(nx,ny,Lx,Ly, "x")	# mesh Lx=Gamma in e_1, Ly in e_2, periodic in x=e_1 dir




V_u = VectorFunctionSpace(mesh, "CG", 2)
V_p = FunctionSpace(mesh, "CG", 1)
V_t = FunctionSpace(mesh, "CG", 1)

Z = V_u * V_p * V_t


upt = Function(Z)
vqs = TestFunction(Z)
upt.assign(0)
u, p, theta = split(upt)
v, q, s = split(vqs)


uOld = Function(V_u)
thetaOld = Function(V_t)



boundary_ids = (1,2)



F_backward = (
	(
		1.0/Pr * inner(u-uOld,v)
		+ tau*(
			 1.0/Pr *inner(dot(u, nabla_grad(u)), v)
			- p*div(v)
			+ nu * inner(grad(u), grad(v))
			- Ra * theta*v[1]
		)
		+ div(u) * q
		+ (theta-thetaOld) * s
		+ tau*( 
			kappa * inner(grad(theta), grad(s))
#			+ inner(dot(u, grad(theta)), s)
			- inner(u*theta, grad(s))
		)
		
	)*dx
	+
	tau * alpha *
	( 
		nu * inner(u[0],v[0])
	)*ds(boundary_ids)
)


F_cranknichel = (
	(
		1.0/Pr * inner(u-uOld,v)
		+ tau*(
			 1.0/Pr * 1.0/2.0*(inner(dot(u, nabla_grad(u)), v)+inner(dot(uOld, nabla_grad(uOld)), v))
			- p*div(v)
			+ nu * 1.0/2.0*(inner(grad(u), grad(v))+inner(grad(uOld), grad(v)))
			- Ra * 1.0/2.0*(inner(theta,v[1]) + inner(thetaOld,v[1]))
		)
		+ inner(div(u),q)
		+ inner(theta-thetaOld,s)
		+ tau*( 
			kappa * 1.0/2.0*(inner(grad(theta), grad(s))+inner(grad(thetaOld), grad(s)))
			+ 1.0/2.0*(inner(dot(u,grad(theta)),s)+inner(dot(uOld,grad(thetaOld)),s))
#			- 1.0/2.0*(inner(u*theta, grad(s))+inner(uOld*thetaOld, grad(s)))
		)
	)*dx
	+
	tau * alpha * 
	( 
		nu * 1.0/2.0*inner(u[0]+uOld[0],v[0])
	)*ds(boundary_ids)
)


F=F_cranknichel

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True),Z.sub(2)])

x,y = SpatialCoordinate(mesh)
scale = 0.1
middleX = Lx*0.5
middleY = Ly*0.5

# initial conditions for u
u = project(as_vector([Constant(0), Constant(0)]), V_u)


# initial conditions for theta
if problem == "boussinesq":
#	theta = project(sin(2*pi*(x/Lx+0.1))*exp(-5*pow(y-middleY,2)), V_t)
#	freq = 1
#	theta = project((sin(freq*2*pi*(x/Lx+y/Ly))*sin(freq*2*pi*(y/Ly))), V_t)
	theta = project(sin(2*pi*(x/Lx+0.1))*exp(-(5*(y-middleY)*(y-middleY))), V_t)
elif problem == "rayleighBenard":
	sinAmp = 0.025
	
	fractionN0 = 1.0/25.0		# after what fraction of height should theta be 10^(-3)
	decayRatioN0 = round(ny*fractionN0) # after n0 gridpoints should be close to 0 -> 10^{-3}		exp(-(2,5^2)) ~ 10^(-3)
	decayRatio = 2.5*ny/(decayRatioN0*Ly)	# decay at the boundary (40 works pretty good for ny = 50)
	frequenceModes = 3
	lowestMode = 2
	#sinAmpList = [0.001,-0.001,0.003,-0.007,0.001,0.006,-0.009,0.001,0.002,0.007]
	#cosAmpList = [0.001,0.001,-0.003,0.007,0.001,-0.006,0.009,0.001,0.002,-0.007]
	sinAmpList = sinAmp*np.ones(frequenceModes)
	cosAmpList = sinAmp*np.ones(frequenceModes)
#	sinAmpList = sinAmp*np.zeros(frequenceModes)
#	cosAmpList = sinAmp*np.zeros(frequenceModes)
	sumOfCos = 0
	sumOfSin = 0
	for k in range(frequenceModes):
		sumOfCos = sumOfCos + cosAmpList[k]*cos((lowestMode+k)*pi*x/Lx)
		sumOfSin = sumOfSin + sinAmpList[k]*sin((lowestMode+k)*pi*x/Lx)
	theta = project(-(1+sumOfCos)*exp(-pow(decayRatio*(y-Ly),2))+(1+sumOfSin)*exp(-pow(decayRatio*y,2)), V_t)
#	theta = project(Constant(0), V_t)
	


uOld.assign(u)
thetaOld.assign(theta)

bc_nonPenetration = DirichletBC(Z.sub(0).sub(1), Constant(0.0), boundary_ids)
bcs = [bc_nonPenetration]

if problem == "rayleighBenard":
	bc_thetaRB_top = DirichletBC(Z.sub(2), -1.0, [2])
	bc_thetaRB_bot = DirichletBC(Z.sub(2), 1.0, [1])
	bcs.append(bc_thetaRB_top)
	bcs.append(bc_thetaRB_bot)

	

problem = NonlinearVariationalProblem(F, upt, bcs=bcs)


# taken from the firedrake rayleigh benard example problem
#solver_params = DON'T USE THE FIREDRAKE RAYLEIGH BÉNARD EXAMPLE PROBLEM SOLVER PARAMETERS, THEY PRODUCE ERRORS
#solver = NonlinearVariationalSolver(problem, nullspace = nullspace, solver_parameters = solver_params)


solver = NonlinearVariationalSolver(problem, nullspace = nullspace)


scriptCopyPath = outputFolder + "used_script_simulation_"+simulationId+".py"
copy(os.path.realpath(__file__), scriptCopyPath)


def projectAvgFree(f):
	#timestarpro = datetime.datetime.now()
	avgF = 1/(Lx*Ly)*assemble(f*dx)
	g = f-avgF
	#print("projecting to average free took ",datetime.datetime.now()-timestarpro)
	#print("zero avg p has measure ",assemble(g*dx))
	return g
	
outFile = File(dataFolder+"test.pvd")
lastWrittenOutput = -1
t = 0


#nyOut = 100
#nxOut = round(nyOut*Lx/Ly)
#meshOut = PeriodicRectangleMesh(nyOut,nxOut,Lx,Ly, "x")
#V_uOut = VectorFunctionSpace(meshOut, "CG", 2)
#V_pOut = FunctionSpace(meshOut, "CG", 1)
#V_tOut = FunctionSpace(meshOut, "CG", 1)
V_uOut = V_u
V_pOut = V_p
V_tOut = V_t




def writeMeshFunctions():
	global lastWrittenOutput
	u.rename("u")
	p.rename("p")
	theta.rename("theta")
	pOutput = p
	if projectPoutputToAverageFree:
		pOutput = projectAvgFree(p)
	outFile.write(u, p, theta, time=t)
	lastWrittenOutput = t


# doesnt matter but otherwise renaming doesnt work
p = Function(V_p)
writeMeshFunctions()
tWorld = datetime.datetime.now()
print(tWorld, "starting to solve")
while(t<tEnd):
	
	solver.solve()
	t = t + tau
	u, p, theta = upt.split()
	u.rename("u")
	p.rename("p")
	theta.rename("theta")
	uOld.assign(u)
	thetaOld.assign(theta)
	
	if round(t,12) >= round(lastWrittenOutput + writeOutputEvery,12):
		writeMeshFunctions()
	
	print(round(t/tEnd*100,12),"% done (after",datetime.datetime.now()-tWorld,"), t=",round(t,12))
	tWorld = datetime.datetime.now()
	
print("completely done at", datetime.datetime.now()," after ",(datetime.datetime.now()-time_start))
