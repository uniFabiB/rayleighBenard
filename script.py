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
import argparse
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
	global infoString
	print("\n")
	errorTime = datetime.datetime.now()
	completeDuration = errorTime - time_start
	print(errorTime,"Exception after",completeDuration)
	print("\n")
	
	infoString += "\n\nERROR INFO"
	infoString += "\nscript stopped because of an error"
	putInfoInInfoString("errorTime", errorTime)
	putInfoInInfoString("completeDuration", completeDuration)
	
	putInfoInInfoString("error type", type)
	putInfoInInfoString("error value", value)
	putInfoInInfoString("error traceback", traceback.format_exception(type, value, tb))
	
	writeInfoFile()
	sys.__excepthook__(type, value, tb)

sys.excepthook = err


print(datetime.datetime.now(),"import complete")
simulationId = str(time_start) +str("_-_") +str(int(np.round(10000000000*np.random.rand())))
simulationId = simulationId.replace(":","-")
simulationId = simulationId.replace(" ","_")
infoString = "simulation info"

def putInfoInInfoString(identifier, value):
	global infoString
	infoString += "\n\t"+identifier+" = \t\t"+str(value)


recoveryScriptFile = "used_script_simulation_"+simulationId+".py"

putInfoInInfoString("simulationId", simulationId)
putInfoInInfoString("recoveryScriptFile",recoveryScriptFile)
putInfoInInfoString("time_start", time_start)



### PARAMETERS ###
n = 100
tau = 0.000005

Lx = 1.0
Ly = 1.0
tEnd = 0.003
alpha = 0.0			# alpha = 1/L_s in navier slip
nu = 1.0			# ... - nu * Laplace u ...
kappa = 1.0			# ... - kappa * Laplace theta ...
Ra = 10**8			# ... + Ra * theta * e_2
Pr = 1.0			# 1/Pr*(u_t+u cdot nabla u) + ...

problem = ""		# either 
					#	boussinesq (no boundary for theta only ic) 
					#	rayleighBenard (boundary -1, 1, ic = 0 + match boundary)
					# 	or change by script argument

timeDiscretisation = "crankNicolson"		# either
						# 	crankNicolson
						# or
						#	backwardEuler

### PARAMETERS END ###


### OPTIONS ###

projectPoutputToAverageFree = True 			# force the output function of p to be average free (doesn't change the calculation)

writeOutputEvery = 0.000025

outputFolder = "output/"
dataFolder = outputFolder + "data/"
infoFilePath = outputFolder + "infoFile.txt"


nx = round(n*Lx/Ly)
ny = n

# managing script arguments
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", "-a", type=float, help="set alpha=1/L_s")
problemArgumentGroup = parser.add_mutually_exclusive_group()
problemArgumentGroup.add_argument("--rayleighBenard", "-rb", action='store_true', help="simulate Rayleigh-Bénard problem (temperature fixed boundary condition, (almost) no initial condition)")
problemArgumentGroup.add_argument("--boussinesq", "-bou", action='store_true', help="simulate Boussinesq problem (arbitrary temperature at boundary, initial condition prescribed)")
args = parser.parse_args()
if args.alpha:
	alpha = float(args.alpha)
	print("alpha set by script argument to", alpha)
if args.rayleighBenard:
	problem = "rayleighBenard"
	print("problem set by script argument to", problem)
elif args.boussinesq:
	problem = "boussinesq"
	print("problem set by script argument to", problem)
	
### OPTIONS END ###

putInfoInInfoString("nx", nx)
putInfoInInfoString("ny", ny)
putInfoInInfoString("Lx", Lx)
putInfoInInfoString("Ly", Ly)
putInfoInInfoString("tau", tau)
putInfoInInfoString("tEnd", tEnd)
if args.alpha:
	putInfoInInfoString("alpha (changed by script argument)", alpha)
else:
	putInfoInInfoString("alpha", alpha)
putInfoInInfoString("nu", nu)
putInfoInInfoString("kappa", kappa)
putInfoInInfoString("Ra", Ra)
putInfoInInfoString("Pr", Pr)
if args.rayleighBenard or args.boussinesq:
	putInfoInInfoString("problem (changed by script argument)", problem)
else:
	putInfoInInfoString("problem", problem)
putInfoInInfoString("timeDiscretisation", timeDiscretisation)

# delete content of infofile
infoFile = open(infoFilePath,"w")
infoFile.write("")
infoFile.close()

def writeInfoFile():
	global infoString
	infoFile = open(infoFilePath,"a")
	infoFile.write(infoString)
	infoFile.close()
	infoString = ""
writeInfoFile()


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



#boundary_ids = (1,2)
boundary_top = 2
boundary_bot = 1
boundary_ids = (boundary_top, boundary_bot)



alpha = Constant(float(alpha))
nu = Constant(float(nu))
kappa = Constant(float(kappa))
Ra = Constant(float(Ra))
Pr = Constant(float(Pr))




F_backwardEuler = (
	(
		1.0/Pr * inner(u-uOld,v)
		+ tau*(
			 1.0/Pr *inner(dot(u, nabla_grad(u)), v)
			+ inner(grad(p),v)
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
	tau * nu * (inner(grad(u)[1,1], v[1])*ds(boundary_bot)
			-inner(grad(u)[1,1], v[1])*ds(boundary_top))
)

F_crankNicolson = (
	(
		1.0/Pr * inner(u-uOld,v)
		+ tau*(
			 1.0/Pr * 1.0/2.0*(inner(dot(u, nabla_grad(u)), v)+inner(dot(uOld, nabla_grad(uOld)), v))
			+ inner(grad(p),v)
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
	tau * nu * 1.0/2.0*(inner(grad(u)[1,1]+grad(uOld)[1,1], v[1])*ds(boundary_bot)
				-inner(grad(u)[1,1]+grad(uOld)[1,1], v[1])*ds(boundary_top))
)

if alpha != 0.0:
	F_backwardEuler = F_backwardEuler + tau * alpha * nu * inner(u[0],v[0])*ds(boundary_ids)
	F_crankNicolson = F_crankNicolson + tau * alpha * nu * 1.0/2.0*inner(u[0]+uOld[0],v[0])*ds(boundary_ids)


if timeDiscretisation == "crankNicolson":
	F=F_crankNicolson
if timeDiscretisation == "backwardEuler":
	F=F_backwardEuler

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
else:
	sys.exit("problem not specified (neither rayleighBenard nor boussinesq)")
	


uOld.assign(u)
thetaOld.assign(theta)

bc_nonPenetration = DirichletBC(Z.sub(0).sub(1), Constant(0.0), boundary_ids)
bcs = [bc_nonPenetration]

if problem == "rayleighBenard":
	bc_thetaRB_top = DirichletBC(Z.sub(2), -1.0, boundary_top)
	bc_thetaRB_bot = DirichletBC(Z.sub(2), 1.0, boundary_bot)
	bcs.append(bc_thetaRB_top)
	bcs.append(bc_thetaRB_bot)

	

problem = NonlinearVariationalProblem(F, upt, bcs=bcs)


# taken from the firedrake rayleigh benard example problem
#solver_params = DON'T USE THE FIREDRAKE RAYLEIGH BÉNARD EXAMPLE PROBLEM SOLVER PARAMETERS, THEY PRODUCE ERRORS
#solver = NonlinearVariationalSolver(problem, nullspace = nullspace, solver_parameters = solver_params)


solver = NonlinearVariationalSolver(problem, nullspace = nullspace)


recoveryScriptPath = outputFolder + recoveryScriptFile
copy(os.path.realpath(__file__), recoveryScriptPath)


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


def projectAvgFree(f, fOutput):
	#timestarprojection = datetime.datetime.now()
	avgF = 1/(Lx*Ly)*assemble(f*dx)
	fOutput.assign(f-avgF)
	#print("projecting to average free took ",datetime.datetime.now()-timestarprojection)
	return fOutput
	

def writeMeshFunctions():
	global lastWrittenOutput
	u.rename("u")
	p.rename("p")
	theta.rename("theta")
	pOutput = p
	if projectPoutputToAverageFree:
		pOutput = projectAvgFree(p, pOutput)
	outFile.write(u, pOutput, theta, time=t)
	lastWrittenOutput = t


# doesnt matter but otherwise renaming doesnt work
p = Function(V_p)
writeMeshFunctions()

infoString += "\n\nsolving info"

tWorld = datetime.datetime.now()
print(tWorld, "starting to solve")
putInfoInInfoString("solving start", tWorld)

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

finishTime = datetime.datetime.now()
completeDuration = finishTime - time_start
putInfoInInfoString("finishTime", finishTime)
putInfoInInfoString("completeDuration", completeDuration)
writeInfoFile()

print("completely done at", finishTime," after ",completeDuration)
