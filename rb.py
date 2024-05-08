import os
os.environ["OMP_NUM_THREADS"] = "1"


import myUtilities
from firedrake import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)		# ignores deprecation warning since firedrake is not updated and the terminal is spammed by 
										#DeprecationWarning: Expr.ufl_domain() is deprecated, please use extract_unique_domain(expr) instead.
										#  warnings.warn("Expr.ufl_domain() is deprecated, please 
import datetime
import weakref

import argparse

outputFolder = "output/"

my_ensemble = Ensemble(COMM_WORLD, COMM_WORLD.size)
comm = my_ensemble.comm

utils = myUtilities.utils(comm, outputFolder)
utils.generateRecoveryScript(__file__)



argParser = argparse.ArgumentParser()
argParser.add_argument("--load", help="load checkpoint file X at time Y (usage: python3 file.py --load checkpoint 100 2.7, this will load checkpoint_mesh.h5 and checkpoint_100.h5 with start time 2.7)", nargs=3)
argParser.add_argument("--refine", help="load checkpoint file X and refine the mesh and functions according to those here and save as checkpoint (usage: python3 file.py --refine checkpoint 100, this will load checkpoint_mesh.h5 and checkpoint_100.h5)", nargs=2)
args = argParser.parse_args()

### PARAMETERS ###
nXY = 512
order = 1

nOut = nXY

uSpace = "Hdiv"			# either Hdiv or Lag

#dt = 0.0001
dt = 0.01
writeOutputEveryXsteps = 10
writeUP = False						# output u and p? False True

writeCheckpointEveryXsteps = 50


Lx = 2.0
Ly = 1.0

t = 0.0
tEnd = 10000 #1.0

### only nav slip ###
alpha = Constant(10.0**0)		# alpha in tau Du n + alpha tau u = 0

				
nu = 1.0			# ... - nu * Laplace u ...
kappa = 1.0			# ... - kappa * Laplace theta ...
Ra = 10.0**7			# ... + Ra * theta * e_2
Pr = 1.0			# 1/Pr*(u_t+u cdot nabla u) + ...


### PARAMETERS END ###


### OPTIONS ###
projectPoutputToAverageFree = False 	# force the output function of p to be average free (doesn't change the calculation)

dataFolder = outputFolder + "data/"


nx = round(nXY*Lx/Ly)
ny = nXY

nxOut = round(nOut*Lx/Ly)
nyOut = nOut

printErrors = True
### OPTIONS END ###


#nCells = nx*ny*2	for diagonal "left" or "right"
#nCells = nx*ny*4	for diagonal "crossed"

utils.putInfoInInfoString("nXY",nXY)
utils.putInfoInInfoString("dt",dt)
utils.putInfoInInfoString("Lx",Lx)
utils.putInfoInInfoString("Ly",Ly)
utils.putInfoInInfoString("tEnd",tEnd)
utils.putInfoInInfoString("kappa",kappa)
utils.putInfoInInfoString("Ra",Ra)
utils.putInfoInInfoString("projectPoutputToAverageFree",projectPoutputToAverageFree)
utils.putInfoInInfoString("dataFolder",dataFolder)
utils.putInfoInInfoString("args",args)




### mesh ###
mesh = PeriodicRectangleMesh(nx,ny,Lx,Ly, "x", comm = comm, diagonal = "crossed", name="myMesh")	# mesh Lx=Gamma in e_1, Ly in e_2, periodic in x=e_1 dir
boundary_id_bot = 1
boundary_id_top = 2
boundary_ids = (1,2)
# change y variable
Vc = mesh.coordinates.function_space()
x, y = SpatialCoordinate(mesh)
# top
ampTop = 0.02
freqTop = 2
freqSinTop = 2
freqCosTop = 6
offsetTop = 0
# bot
ampBot = ampTop
freqBot = 1
freqSinBot = 3
freqCosBot = 8
offsetBot = 0.5*Lx
# top
y = y + y/Ly * ampTop * sin(2*pi*freqTop*(x-offsetTop)/Lx+freqSinTop*sin(2*pi*(x-offsetTop)/Lx)+freqCosTop*cos(2*pi*(x-offsetTop)/Lx))
# bot
y = y + (1-y/Ly) * ampBot * sin(2*pi*freqBot*(x-offsetBot)/Lx+freqSinBot*sin(2*pi*(x-offsetBot)/Lx)+freqCosBot*cos(2*pi*(x-offsetBot)/Lx))
f = Function(Vc).interpolate(as_vector([x, y]))
mesh.coordinates.assign(f)

nPerCore = abs(sqrt(mesh.num_entities(2)/(4)))  # seems to work but not super accurate
utils.print("sqrt(n^2 / core) ", nPerCore)
#mesh = Mesh('mesh1.msh')


if args.refine:
	filename = args.refine[0]
	step = int(args.refine[1])
	if COMM_WORLD.size > 1:
		utils.print("!WARNING! running in parallel, but refinement needs to be on a single process")
	utils.print("refining filename"+ str(step) + ".h5 to n="+str(nXY))
	with CheckpointFile(args.refine[0] + "_mesh.h5", 'r') as meshInFile:
		oldMesh = meshInFile.load_mesh("myMesh")
	V_tOld = FunctionSpace(oldMesh, "CG", order)
	V_t = FunctionSpace(mesh, "CG", order)
	thetaOld = Function(V_tOld, name="theta")
	theta = Function(V_t, name="theta")
	theta.assign(0)
	with CheckpointFile(filename + "_" + str(step) + ".h5", 'r') as inFile:
		uOld = inFile.load_function(oldMesh, "u")
		thetaOld = inFile.load_function(oldMesh, "theta")
	coordsFine = Function(VectorFunctionSpace(mesh, "CG", 1)).interpolate(SpatialCoordinate(mesh))
	uOld_values = np.array(uOld.at(coordsFine.dat.data))
	thetaOld_values = np.array(thetaOld.at(coordsFine.dat.data))
	theta.dat.data[:] = thetaOld_values
	with CheckpointFile(outputFolder + "checkpoint_refined_mesh.h5", 'w') as meshOutFile:
		utils.print("writing mesh checkpoint file")
		meshOutFile.save_mesh(mesh)
		
	if uSpace == "Hdiv":
		V_u = FunctionSpace(mesh, "RT", order+1)
	elif uSpace == "Lag":
		V_u = VectorFunctionSpace(mesh, "CG", order+1)
	V_p = FunctionSpace(mesh, "CG", order)
	V_t = FunctionSpace(mesh, "CG", order)
	u = Function(V_u, name="u")
	u.assign(as_vector([0,0]))
	with CheckpointFile(outputFolder + "checkpoint_refined_"+ str(step) + ".h5", 'w') as outFile:
		utils.print("writing functions checkpoint file")
		u.rename("u")
		outFile.save_function(u)
		theta.rename("theta")
		outFile.save_function(theta)
	raise Exception("done refining")
	

if args.load:
	with CheckpointFile(args.load[0] + "_mesh.h5", 'r') as meshInFile:
		mesh = meshInFile.load_mesh("myMesh")

with CheckpointFile(outputFolder + "checkpoints/" + "checkpoint_mesh.h5", 'w') as meshOutFile:
	utils.print("writing mesh checkpoint file")
	meshOutFile.save_mesh(mesh)

utils.writeInfoFile()



n = FacetNormal(mesh)
tau = as_vector((-n[1],n[0]))

if uSpace == "Hdiv":
	V_u = FunctionSpace(mesh, "RT", order+1)
elif uSpace == "Lag":
	V_u = VectorFunctionSpace(mesh, "CG", order+1)
V_p = FunctionSpace(mesh, "CG", order)
V_t = FunctionSpace(mesh, "CG", order)


#V_u = FunctionSpace(mesh, "BDM", 1)
#V_p = FunctionSpace(mesh, "DG", 0)
#V_t = FunctionSpace(mesh, "CG", 1)




#alpha = Function(V_p,name="alpha").interpolate(conditional(x<Lx/2.0, 0.001, 1000.0))
#alphaFile = File(dataFolder+"alpha.pvd", comm = comm).write(alpha)





Z = V_u * V_p * V_t

upt = Function(Z, name="upt")
vqs = TestFunction(Z)
upt.assign(0)
u, p, theta = split(upt)
v, q, s = split(vqs)


uOld = Function(V_u)
thetaOld = Function(V_t)

nu = Constant(float(nu))
kappa = Constant(float(kappa))
Ra = Constant(float(Ra))
Pr = Constant(float(Pr))

Du = 0.5*(grad(u)+nabla_grad(u))
DuOld = 0.5*(grad(uOld)+nabla_grad(uOld))
Dv = 0.5*(grad(v)+nabla_grad(v))
v_n = dot(n,v)*n
v_tau = dot(tau,v)*tau
u_n = dot(n,u)*n
u_tau = dot(tau,u)*tau


def createCheckpoint():
	global lastWrittenCheckpoint
	utils.print("creating checkpoint at step " + str(step))
	with CheckpointFile(outputFolder + "checkpoints/" + "checkpoint_"+ str(step) + ".h5", 'w') as outFile:
		u.rename("u")
		outFile.save_function(u)
		theta.rename("theta")
		outFile.save_function(theta)
	lastWrittenCheckpoint = step

	
	
	

# u_t + u cdot nabla u + nabla p - sqrt(Pr/Ra) Delta u = theta e_2
# theta_t + u cdot nabla theta - 1/sqrt(Pr*Ra) Delta theta = 0
# Padberg-Gehle reformulation -> automatic free fall time
# seems to yield any improvement over the other method (except automatic rescaling of time)
# doesn't converge for Ra 10^8 and n = 128 after some time



F_crankNicolson_freeFall = (
	inner(u-uOld,v)*dx
	+ dt*(
		1.0/2.0*(inner(dot(u, nabla_grad(u)), v)+inner(dot(uOld, nabla_grad(uOld)), v))*dx
		+ inner(grad(p),v)*dx
		+ sqrt(Pr/Ra) * nu * 1.0/2.0*(inner(grad(u), grad(v))+inner(grad(uOld), grad(v)))*dx
		- 1.0/2.0*(inner(theta,v[1]) + inner(thetaOld,v[1]))*dx
	)
	+ inner(theta-thetaOld,s)*dx
	+ dt*( 
		1.0/2.0*(inner(dot(u,grad(theta)),s)+inner(dot(uOld,grad(thetaOld)),s))*dx
		+ 1.0/(2.0*sqrt(Pr*Ra)) * kappa * (inner(grad(theta), grad(s))+inner(grad(thetaOld), grad(s)))*dx
		- 1.0/(2.0*sqrt(Pr*Ra)) * kappa * (inner(dot(n,grad(theta)),s)+inner(dot(n,grad(thetaOld)),s))*ds
	)
	+ inner(u,grad(q))*dx
)

#https://gmd.copernicus.org/preprints/gmd-2021-367/gmd-2021-367.pdf


F_crankNicolson_freeFall_NavSlip_hDiv = (
	inner(u-uOld,v)*dx
	+ dt*(
		1.0/2.0*inner(dot(u, nabla_grad(u)) + dot(uOld, nabla_grad(uOld)), v)*dx
		+ sqrt(Pr/Ra) * nu * inner(Du + DuOld, Dv)*dx
		+ sqrt(Pr/Ra) * nu * inner(alpha * (u + uOld),v)*ds
		- 1.0/2.0*inner(theta+thetaOld,v[1])*dx
	)
	+ inner(theta-thetaOld,s)*dx
	+ dt*( 
		1.0/2.0*inner(dot(u,grad(theta))+dot(uOld,grad(thetaOld)),s)*dx
		+ 1.0/(2.0*sqrt(Pr*Ra)) * kappa * inner(grad(theta+thetaOld), grad(s))*dx
		- 1.0/(2.0*sqrt(Pr*Ra)) * kappa * inner(dot(n,grad(theta+thetaOld)),s)*ds #term?!?!?!?!?!?!?!
	)
	+ inner(u,grad(q))*dx
	+ inner(grad(p),v)*dx
)


F_back_freeFall_NavSlip_hDiv = (
	inner(u-uOld,v)*dx
	+ dt*(
		1.0/1.0*(inner(dot(u, nabla_grad(u)), v))*dx
		+ 2.0 * sqrt(Pr/Ra) * nu *(inner(Du, Dv))*dx
		+ 2.0 * sqrt(Pr/Ra) * nu * alpha * (inner(u,v))*ds
		#+ 2.0 * sqrt(Pr/Ra) * nu * (inner(dot(v,n),dot(n,dot(Du,n))))*ds
		#+ 2.0 * sqrt(Pr/Ra) * nu * (inner(dot(u,n),dot(n,dot(Dv,n))))*ds
		- 1.0/1.0*(inner(theta,v[1]))*dx
	)
	+ inner(theta-thetaOld,s)*dx
	+ dt*( 
		1.0/1.0*(inner(dot(u,grad(theta)),s))*dx
		+ 1.0/(1.0*sqrt(Pr*Ra)) * kappa * inner(grad(theta), grad(s))*dx
		- 1.0/(1.0*sqrt(Pr*Ra)) * kappa * inner(dot(n,grad(theta)),s)*ds #term?!?!?!?!?!?!?!
	)
	+ inner(u,grad(q))*dx
	+ inner(grad(p),v)*dx
)

def perp(f):
	return as_vector([-f[1],f[0]])
# Hdiv nonlinear term
# 1/2 grad (u^2) - u^perp nabla cdot u^perp = (u cdot nabla) u
grad_term = - 1.0/2.0 * inner(div(v),dot(u,u))*dx
perp_term = - inner(v,div(perp(u))*perp(u))*dx
nonlin_term = (
	grad_term
	+ perp_term
)

gamma = Constant((100.0))
c = Constant(10**1.0) # dark magic

nonlin_term_cn = (
	- 1.0/4.0 * inner(div(v),dot(u,u)+dot(uOld,uOld))*dx
	- 1.0/2.0 * inner(v,div(perp(u))*perp(u)+div(perp(uOld))*perp(uOld))*dx
)

viscous_term_cn = (
	1.0/2.0 * inner(grad(u+uOld), grad(v))*dx #this is the term over omega from the integration by parts
	+ inner(avg(outer(v,n)),avg(grad(u+uOld)))*dS #this the term over interior surfaces from integration by parts
	+ inner(avg(outer(u+uOld,n)),avg(grad(v)))*dS
	+ alpha*inner(u+uOld,v)*ds #This deals with boundaries
)



F_hDiv_int_cn = (
	inner(u-uOld,v)*dx
	+ dt*(
		nonlin_term_cn
		+ sqrt(Pr/Ra) * nu * viscous_term_cn
		- 1.0/2.0*inner(theta+thetaOld,v[1])*dx
	)
	+ inner(theta-thetaOld,s)*dx
	+ dt*( 
		1.0/2.0*inner(dot(u,grad(theta))+dot(uOld,grad(thetaOld)),s)*dx
		+ 1.0/(2.0*sqrt(Pr*Ra)) * kappa * inner(grad(theta+thetaOld), grad(s))*dx
		- 1.0/(2.0*sqrt(Pr*Ra)) * kappa * inner(dot(n,grad(theta+thetaOld)),s)*ds #term?!?!?!?!?!?!?!
	)
	- inner(p,div(v))*dx
	+ inner(div(u),q)*dx
	
#	+ gamma*div(v)*div(u)*dx		# stabilizing term 1
	+ c*inner(jump(v),jump(u))*dS		# stabilizing term 2
	+ c*inner(jump(v),jump(uOld))*dS	# stabilizing term 3
	
)

# Hdiv with interior
#dealing with viscous term
viscous_byparts1 = inner(grad(u), grad(v))*dx #this is the term over omega from the integration by parts
viscous_byparts2 = 2*inner(avg(outer(v,n)),avg(grad(u)))*dS #this the term over interior surfaces from integration by parts
viscous_symetry = 2*inner(avg(outer(u,n)),avg(grad(v)))*dS #this the term ensures symetry while not changing the continuous equation
viscous_stab = c*inner(jump(v),jump(u))*dS #stabilizes the equation, somehow
#viscous_byparts2_ext = (inner(outer(v,n),grad(u)) + inner(outer(u,n),grad(v)))*ds #This deals with boundaries TOFIX : CONSIDER NON-0 BDARIEs 
viscous_byparts2_ext = 2*alpha*inner(u,v)*ds #This deals with boundaries
#viscous_ext = c*nXY*inner(v,u)*ds #this is a penalty term for the boundaries

viscous_terms = (
	viscous_byparts1
	 + viscous_byparts2
	 + viscous_symetry
#	 + viscous_stab			#used directly in the variational form to omit Ra scaling for it
	 + viscous_byparts2_ext
#	 + viscous_ext
)

F_hDiv_int_back = (
	inner(u-uOld,v)*dx
	+ dt*(
		nonlin_term
		+ 1.0 * sqrt(Pr/Ra) * nu *viscous_terms
		- 1.0/1.0*(inner(theta,v[1]))*dx
	)
	+ inner(theta-thetaOld,s)*dx
	+ dt*( 
		1.0/1.0*(inner(dot(u,grad(theta)),s))*dx
		+ 1.0/(1.0*sqrt(Pr*Ra)) * kappa * inner(grad(theta), grad(s))*dx
		- 1.0/(1.0*sqrt(Pr*Ra)) * kappa * inner(dot(n,grad(theta)),s)*ds #term?!?!?!?!?!?!?!
	)
	- inner(div(u),q)*dx
	- inner(p,div(v))*dx
	
	+ viscous_stab
#	+ gamma*div(v)*div(u)*dx
)

F = F_crankNicolson_freeFall_NavSlip_hDiv


# initial conditions for u
u = project(Constant([0.0,0.0]), V_u)

x, y = SpatialCoordinate(mesh)
#icTheta = 0.5*sin(2*pi*x/Lx)*exp(-4*(0.5-y)**2)+0.5


#icTheta = 10.0*sin(2*pi*x/Lx)*sin(2*pi*y/Ly)*exp(-1*(x**2+y**2))
expFactor = 0.5
[amp, x0, y0] = [10.0, -1.0, 2.0]
icTheta = amp*exp(-expFactor*((x-x0)**2+(y-y0)**2))
[amp, x0, y0] = [-10.0, 2.0, 2.0]
icTheta = icTheta + amp*exp(-expFactor*((x-x0)**2+(y-y0)**2))
[amp, x0, y0] = [-10.0, -3.0, 2.0]
icTheta = icTheta + amp*exp(-expFactor*((x-x0)**2+(y-y0)**2))
[amp, x0, y0] = [10.0, 1.0, 1.0]
icTheta = icTheta + amp*exp(-expFactor*((x-x0)**2+(y-y0)**2))
[amp, x0, y0] = [10.0, 6.0, 3.0]
icTheta = icTheta + amp*exp(-expFactor*((x-x0)**2+(y-y0)**2))
[amp, x0, y0] = [-10.0, 4.0, 0.0]
icTheta = icTheta + amp*exp(-expFactor*((x-x0)**2+(y-y0)**2))

icTheta = Constant(0.5)
theta = project(icTheta, V_t)

step = 0

if args.load:
	filename = args.load[0]
	step = int(args.load[1])
	t = float(args.load[2])
	utils.print("loading step " + str(step) + " from file " + filename + "_" + str(step) + ".h5 at time " + str(t))
	with CheckpointFile(filename + "_" + str(step) + ".h5", 'r') as inFile:
		u = inFile.load_function(mesh, "u")
		theta = inFile.load_function(mesh, "theta")

uOld.assign(u)
thetaOld.assign(theta)

bcs = []

bc_rbBot = DirichletBC(Z.sub(2), Constant(1.0), (boundary_id_bot))
bc_rbTop = DirichletBC(Z.sub(2), Constant(0.0), (boundary_id_top))
bcs.append(bc_rbBot)
bcs.append(bc_rbTop)


u_n = dot(u,n)
v_n = dot(v,n)


if uSpace == "Hdiv":
	bc_noPenHdiv = DirichletBC(Z.sub(0), Constant((0.0,0.0)), "on_boundary") # because of the hdiv space setting 0 bc for it is only setting u cdot n = 0 #https://github.com/firedrakeproject/firedrake/issues/169#issuecomment-34557942
	bcs.append(bc_noPenHdiv)
elif uSpace == "Lag":
	bcs_noSlip = DirichletBC(Z.sub(0), Constant((0.0,0.0)), "on_boundary")
	bcs.append(bcs_noSlip)
	

problem = NonlinearVariationalProblem(F, upt, bcs = bcs)


nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True,comm=comm), Z.sub(2)])

parameters_my = {
	"mat_type": "matfree",
#	"snes_monitor": None,
#	"ksp_monitor_true_residual": None,
#	"ksp_converged_reason": None,
	
	"ksp_type": "gmres",
	"ksp_gmres_restart": 15,
	"pc_type": "fieldsplit",
	"pc_fieldsplit_type": "multiplicative",
	"pc_fieldsplit_0_fields": "0,1",
	"pc_fieldsplit_1_fields": "2",

	"fieldsplit_0": {
		"ksp_type": "preonly",
		"pc_type": "python",
		"pc_python_type": "firedrake.AssembledPC",
		"assembled_pc_type": "lu",
		"assembled_pc_factor_mat_solver_type": "mumps",
		},

	"fieldsplit_1": {
		"ksp_type": "preonly",
		"pc_type": "python",
		"pc_python_type": "firedrake.AssembledPC",
		"assembled_pc_type": "lu",
	}
}
appctx = {"velocity_space": 0}

solver = NonlinearVariationalSolver(problem, nullspace = nullspace, solver_parameters=parameters_my, appctx=appctx)
#solver = NonlinearVariationalSolver(problem, nullspace = nullspace)

uptFile = File(dataFolder+"upt.pvd", comm = comm)
lastWrittenOutput = step
lastWrittenCheckpoint = step

def projectAvgFree(f, fOutput):
	avgF = 1/(Lx*Ly)*assemble(f*dx)
	fOutput.assign(f-avgF)
	return fOutput


def writeMeshFunctions():
	global lastWrittenOutput
	
	if nxOut != nx or nyOut != ny:
		thetaOut = project(theta, V_ptOut)
	else:
		thetaOut = theta
#		thetaOut = project(theta, V_ptOut)		
	thetaOut.rename("theta")
	if writeUP:
		if nxOut != nx or nyOut != ny:
			uOut = project(u, V_uOut)
			pOut = project(p, V_ptOut)
		else:
			uOut = u
			pOut = p
		if projectPoutputToAverageFree:
			pOut = projectAvgFree(pOut, pOut)
		
		uOut.rename("u")
		pOut.rename("p")
		uptFile.write(uOut, pOut, thetaOut, time=t)
	else:
		uptFile.write(thetaOut, time=t)
	lastWrittenOutput = step


# doesnt matter but otherwise renaming doesnt work
p = Function(V_p)
p = project(Constant(0.0), V_p)


meshOut = mesh
meshOut._parallel_compatible = {weakref.ref(mesh)}

V_uOut = VectorFunctionSpace(meshOut, "CG", 1)
V_ptOut = FunctionSpace(meshOut, "CG", 1)

writeMeshFunctions()

utils.putInfoInInfoString("solving info","")

utils.putInfoInInfoString("solving start", datetime.datetime.now())
COMM_WORLD.Barrier()
utils.print("starting to solve")


tWorld = datetime.datetime.now()

convergencewarningnumber = 0

def calcBdryL2(func):
	return assemble(inner(func,func)*ds)
	
	
def writeFactorError(fac, base):
		error = calcBdryL2(fac*alpha*dot(u,tau)+dot(dot(n,Du),tau))
		#utils.print("temp\t",fac,"\t",error, "\t", (error/base))
		utils.print("temp\t",fac,"\t", (error/base))

iterationTry = 0		# tracks iterates for convergence errors and tries again with different approaches (solverparams, dt)

checkPointIndex = 0


initial_dt = dt

while(t<=tEnd):
	try:
		#utils.print(solver.parameters)
		solver.solve()
	except ConvergenceError as convError:
		utils.print(convError)
		myErrorMsg = ""
		if "DIVERGED_LINEAR_SOLVE" in str(convError) or "DIVERGED_DTOL" in str(convError) or "DIVERGED_MAX_IT" in str(convError):
			myErrorMsg = ""
			if "DIVERGED_LINEAR_SOLVE" in str(convError):
				myErrorMsg = "DIVERGED_LINEAR_SOLVE"
			if "DIVERGED_DTOL" in str(convError):
				myErrorMsg = "DIVERGED_DTOL"
			if "DIVERGED_MAX_IT" in str(convError):
				myErrorMsg = "DIVERGED_MAX_IT"
				
			if iterationTry == 2:
				utils.print(myErrorMsg)
				raise Exception("Diverged with all fixes")
			if iterationTry == 1:
				myFullErrorMsg = myErrorMsg + ", trying with different dt"
				utils.print(myFullErrorMsg)
				utils.putInfoInInfoString("convergence warning "+str(convergencewarningnumber), " at time " + str(t + dt) + ": " + myFullErrorMsg)
				dt = 0.5*initial_dt				
				iterationTry = 2
			if iterationTry == 0:
				myFullErrorMsg = myErrorMsg + ", trying again with different solver parameters"
				utils.print(myFullErrorMsg)
				utils.putInfoInInfoString("convergence warning "+str(convergencewarningnumber), " at time " + str(t + dt) + ": " + myFullErrorMsg)
				solver = NonlinearVariationalSolver(problem, nullspace = nullspace)
				iterationTry = 1
			convergencewarningnumber += 1
			utils.writeInfoFile()
		else:
			utils.print("error in solve")
			raise Exception("convergence error")
			
	else:
		step += 1
		t = round(t+dt,9)
		utils.setSimulationTime(t)
		u, p, theta = upt.subfunctions	# depending on the firedrake version have to either use upt.split() (old) or upt.subfunctions (newer)
		
		
		uOld.assign(u)
		thetaOld.assign(theta)
		
		if step >= lastWrittenOutput + writeOutputEveryXsteps:
			writeMeshFunctions()
			
		if step >= lastWrittenCheckpoint + writeCheckpointEveryXsteps:
			createCheckpoint()
			
		utils.print(round(t/tEnd*100,9),"% done (after",datetime.datetime.now()-tWorld,"), t=",round(t,9))
		utils.print(" ")
		utils.print("div(u)\t",norm(div(u)))	
		utils.print(" ")
		utils.print("u n\t\t",calcBdryL2(dot(u,n)))
		utils.print("u tau\t",calcBdryL2(dot(u,tau)))	
		utils.print("n Du tau\t",calcBdryL2(dot(n,dot(tau,Du))))
		utils.print("n grad theta\t",calcBdryL2(dot(n,grad(theta))))
		
#		utils.print("temp\t",factor,"\t",assemble(inner(factor*alpha*dot(u,tau)+dot(dot(n,Du),tau),factor*alpha*dot(u,tau)+dot(dot(n,Du),tau))*ds))
		utils.print(" ")
		base = 	calcBdryL2(dot(u,tau)) + calcBdryL2(dot(n,dot(tau,Du)))
#		factor = 0.25
		#writeFactorError(0.25,base)
		writeFactorError(0.50,base)
		#writeFactorError(0.75,base)
		writeFactorError(0.90,base)
		writeFactorError(1.00,base)
		writeFactorError(1.10,base)
		#writeFactorError(1.25,base)
		#writeFactorError(1.50,base)
		#writeFactorError(1.75,base)
		writeFactorError(2.00,base)
		
		# seems to converge to the nav slip for order to infty
		# it should be the true solution if it solves the variational problem
		# at least up to comp errors
		# i guess/hope the var prob is more accurate then the derivative on the boundary
		
		
		utils.print(" ")
		utils.print(" ")
		tWorld = datetime.datetime.now()
		
		
		if iterationTry > 0:
			utils.print("reverting to my parameters")
			iterationTry = 0
			solver = NonlinearVariationalSolver(problem, nullspace = nullspace, solver_parameters=parameters_my, appctx=appctx)
			dt = initial_dt
	
utils.writeEndInfo()
