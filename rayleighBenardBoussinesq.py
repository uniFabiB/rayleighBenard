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
	putInfoInInfoString("errorSimulationTime",t)
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
n = 20
tau = 0.0001

problem = ""
######## TODO set kappa for boussinesq by script ########
Lx = 1.0
Ly = 1.0

t = 0.0
tEnd = 1.0

alpha = 0.0			# alpha = 1/L_s in navier slip
nu = 1.0			# ... - nu * Laplace u ...
kappa = 1.0			# ... - kappa * Laplace theta ...
Ra = 10.0**6			# ... + Ra * theta * e_2
Pr = 1.0			# 1/Pr*(u_t+u cdot nabla u) + ...

problem = ""		# either 
					#	boussinesq (no boundary for theta only ic) 
					#	rayleighBenard (boundary -1, 1, ic = 0 + match boundary)
					# 	or change by script argument

timeDiscretisation = ""		# either
						# 	crankNicolson
						# or
						#	backwardEuler

### PARAMETERS END ###


### OPTIONS ###

projectPoutputToAverageFree = True 			# force the output function of p to be average free (doesn't change the calculation)

writeOutputEvery = 0.0#0.000025

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


#mesh = PeriodicRectangleMesh(nx,ny,Lx,Ly, "x")	# mesh Lx=Gamma in e_1, Ly in e_2, periodic in x=e_1 dir

#mesh = Mesh('mygeo.msh', periodic_coords=(0,1))
mesh = Mesh('mygeo.msh')

boundary_id_bot = 1
boundary_id_top = 2
boundary_ids = (boundary_id_bot, boundary_id_top)


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






alpha = Constant(float(alpha))
nu = Constant(float(nu))
kappa = Constant(float(kappa))
Ra = Constant(float(Ra))
Pr = Constant(float(Pr))









e1 = Constant([1.0,0.0])
e2 = Constant([0.0,1.0])



normal = FacetNormal(mesh)
tangential = as_vector((-normal[1],normal[0]))

F_crankNicolson = (
	(
		1.0/Pr * inner(u-uOld,v)
		+ tau*(
			 1.0/Pr * 1.0/2.0*(inner(dot(u, nabla_grad(u)), v)+inner(dot(uOld, nabla_grad(uOld)), v))
			+ inner(grad(p),v)
			+ nu * 1.0/2.0*(inner(grad(u), grad(v))+inner(grad(uOld), grad(v)))
			- Ra * 1.0/2.0*(inner(theta,v[1]) + inner(thetaOld,v[1]))
		)
		- inner(u,grad(q))	# L_# perp L_{sigma,tau} deswegen implied automatisch dass #u*n = 0
#		+ inner(div(u),q)
		+ inner(theta-thetaOld,s)
		+ tau*( 
			kappa * 1.0/2.0*(inner(grad(theta), grad(s))+inner(grad(thetaOld), grad(s)))
			+ 1.0/2.0*(inner(dot(u,grad(theta)),s)+inner(dot(uOld,grad(thetaOld)),s))
#			- 1.0/2.0*(inner(u*theta, grad(s))+inner(uOld*thetaOld, grad(s)))
		)
	)*dx
#	-
#	- tau * nu * 1.0/2.0*(inner(dot(normal,dot(normal, nabla_grad(u))), dot(v,normal))+inner(dot(normal,dot(normal, nabla_grad(uOld))), dot(v,normal)))*ds(boundary_ids)

)

if alpha != 0.0:
#	F_backwardEuler = F_backwardEuler + tau * alpha * nu * inner(u,v)*ds("on_boundary")
#	F_crankNicolson = F_crankNicolson + tau * alpha * nu * 1.0/2.0*inner(u+uOld,v)*ds(boundary_ids)
	F_crankNicolson = F_crankNicolson + tau * alpha * nu * 1.0/2.0*inner(dot(u+uOld,tangential),dot(v,tangential))*ds(boundary_ids)


F = F_crankNicolson
putInfoInInfoString("timeDiscretisation", timeDiscretisation)
writeInfoFile()

nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True), Z.sub(2)])

x,y = SpatialCoordinate(mesh)
scale = 0.1
middleX = Lx*0.5
middleY = Ly*0.5

# initial conditions for u
u = project(Constant([0.0,0.0]), V_u)


# initial conditions for theta
slope = 10.0
slopeX = slope/Lx
slopeY = slope/Ly
leftPlus = Max(Min(slope*(x-1.0/8.0*Lx),1.0),0.0)
rightPlus = Max(Min(slope*(-(x-3.0/8.0*Lx)),1.0),0.0)
botPlus = Max(Min(slope*(y-1.0/4.0*Ly),1.0),0.0)
topPlus = Max(Min(slope*(-(y-3.0/4.0*Ly)),1.0),0.0)
lrPlus = Min(leftPlus,rightPlus)
tbPlus = Min(botPlus,topPlus)
Plus = lrPlus*tbPlus

leftMinus = Max(Min(slope*(x-5.0/8.0*Lx),1.0),0.0)
rightMinus = Max(Min(slope*(-(x-7.0/8.0*Lx)),1.0),0.0)
botMinus = Max(Min(slope*(y-1.0/4.0*Ly),1.0),0.0)
topMinus = Max(Min(slope*(-(y-3.0/4.0*Ly)),1.0),0.0)
lrMinus = Min(leftMinus,rightMinus)
tbMinus = Min(botMinus,topMinus)
Minus = lrMinus*tbMinus

#theta = project(Plus-Minus, V_t)
theta = project(Constant(0), V_t)



uOld.assign(u)
thetaOld.assign(theta)



# LOOK AT https://prism.ac.uk/wp-content/uploads/2020/09/Beyond-CFD_Koki-Sagiyama.pdf !!!!!!
# also at https://prism.ac.uk/wp-content/uploads/2020/09/Beyond-CFD_Koki-Sagiyama.pdf for n_2 partial_2 u_2 v_2
# doesnt work! maybe his own package
# seems impossible via dirichletbc: https://github.com/firedrakeproject/firedrake/issues/981
# have to do it by variational formulation
#bc_nonPenetration = DirichletBC(Z.sub(0), 
#V1 = BoundaryComponentSubspace(Z.sub(0), "on_boundary", normal)
#bc_nonPenetration = DirichletBC(Z.sub(0), Constant([0.0, 0.0]), "on_boundary")
#bc_nonPenetration = DirichletBC(Z.sub(0).sub(1), Constant(0.0), (boundary_bot,boundary_top))

#bcs = [bc_nonPenetration]

#print(assemble(inner(normal,tangential)*ds))
#File(dataFolder+"nTau.pvd").write(normal, tangential)
#FbcNonPenetration = (inner(dot(u,normal),dot(v,normal))+inner(u[0],v[0]))*ds((boundary_bot,boundary_top))
#bc_nonPenetrationVar = EquationBC(FbcNonPenetration==0, u, (boundary_bot,boundary_top), V=Z.sub(0))
#bcs = [bc_nonPenetrationVar]

#FbcNonPenetration = inner(dot(u,normal),dot(v,normal))*ds((boundary_bot,boundary_top))
#FbcTan = inner(dot(u,tangential),q)*ds((boundary_bot,boundary_top))
#bc_nonPenetration = EquationBC(FbcNonPenetration==0, u, (boundary_bot,boundary_top), V=Z.sub(0))
#bc_tan = EquationBC(FbcTan==0, u, (boundary_bot,boundary_top), V=Z.sub(0))
#bcs = [bc_nonPenetration, bc_tan]



FbcZero = inner(u,v)*ds
bc_zero = EquationBC(FbcZero==0, u, "on_boundary", V=Z.sub(0))

FbcNonPenetration = inner(u,normal*q)*ds
bc_nonPenetration = EquationBC(FbcNonPenetration==0, u, "on_boundary", V=Z.sub(0))

FbcTan = inner(dot(u,tangential),s)*ds()
FbcTanOld = inner(dot(uOld,tangential),dot(v,tangential))*ds()
bc_tan = EquationBC(FbcTan==0, u, "on_boundary", V=Z.sub(0))

Fbc_nDut = 1.0/2.0*inner(dot(dot(nabla_grad(u)+grad(u),normal),tangential),dot(v,tangential))*ds()
Fbc_nDutOld = 1.0/2.0*inner(dot(dot(nabla_grad(uOld)+grad(uOld),normal),tangential),dot(v,tangential))*ds()
#F_navSlip = 1.0/2.0*(Fbc_nDut+Fbc_nDutOld)+1.0/2.0*alpha*(FbcTan+FbcTanOld)
F_navSlip = Fbc_nDut+alpha*FbcTan
F_temp = 1.0/10000000.0*inner(tangential,v)*ds()-FbcTan
bc_temp = EquationBC(F_temp==0, u, "on_boundary", V=Z.sub(0))
bc_navSlip = EquationBC(F_navSlip==0, u, "on_boundary", V=Z.sub(0))
#bcs = [bc_nonPenetration, bc_navSlip]


bcs = []

bc_rbBot = DirichletBC(Z.sub(2), Constant(1), (boundary_id_bot))
bc_rbTop = DirichletBC(Z.sub(2), Constant(-1), (boundary_id_top))
bcs.append(bc_rbBot)
bcs.append(bc_rbTop)

#bcs.append(bc_zero)
#bcs.append(bc_navSlip)
#bcs.append(bc_nonPenetration)

problem = NonlinearVariationalProblem(F, upt, bcs = bcs)


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
	
	print("\n",round(t/tEnd*100,12),"% done (after",datetime.datetime.now()-tWorld,"), t=",round(t,12))
	l2Normal = abs(sqrt(assemble(inner(dot(u,normal),dot(u,normal))*ds)))
#	print("\tl2 normal u ", l2Normal)
	l2Tangential = abs(sqrt(assemble(inner(dot(u,tangential),dot(u,tangential))*ds)))
#	print("\tl2 tangential u ", l2Tangential)
	print("\tratio (l2 normal/l2 tangential) ",round(l2Normal/l2Tangential,12))
	l1DivU = norm(div(u),"l1")
#	print("\tl1 div u ", l1DivU)
	l1U = norm(u,"l1")
#	print("\tl1 u ", l1U)
	print("\tratio (l1 div u/l1 u)",round(l1DivU/l1U,12))
	l2DivU = norm(div(u),"l2")
#	print("\tl2 div u ", l2DivU)
	l2U = norm(u,"l2")
#	print("\tl2 u ", l2U)
	print("\tratio (l2 div u/l2 u)",round(l2DivU/l2U,12))
	tWorld = datetime.datetime.now()

finishTime = datetime.datetime.now()
completeDuration = finishTime - time_start
putInfoInInfoString("finishTime", finishTime)
putInfoInInfoString("completeDuration", completeDuration)
writeInfoFile()

print("completely done at", finishTime," after ",completeDuration)

