import os
os.environ["OMP_NUM_THREADS"] = "1"

import myUtilities
from firedrake import *
import datetime
import weakref

outputFolder = "output/"

my_ensemble = Ensemble(COMM_WORLD, COMM_WORLD.size)
comm = my_ensemble.comm

utils = myUtilities.utils(comm, outputFolder)
utils.generateRecoveryScript(__file__)



### PARAMETERS ###
nXY = 64
order = 1

nOut = nXY

uSpace = "Hdiv"			# either Hdiv or Lag

#dt = 0.0001
dt = 0.1
writeOutputEveryXsteps = 1
writeOutputEvery = dt*writeOutputEveryXsteps			# write mesh functions
writeUP = True						# output u and p? False True



Lx = 2.0
Ly = 1.0

t = 0.0
tEnd = 10000 #1.0

### only nav slip ###
#alpha = 1.0		# alpha in tau Du n + alpha tau u = 0

				
nu = 1.0			# ... - nu * Laplace u ...
kappa = 1.0			# ... - kappa * Laplace theta ...
Ra = 10.0**5			# ... + Ra * theta * e_2
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




### mesh ###
mesh = PeriodicRectangleMesh(nx,ny,Lx,Ly, "x", comm = comm, diagonal = "crossed")	# mesh Lx=Gamma in e_1, Ly in e_2, periodic in x=e_1 dir
boundary_id_bot = 1
boundary_id_top = 2
boundary_ids = (1,2)
# change y variable
Vc = mesh.coordinates.function_space()
x, y = SpatialCoordinate(mesh)
# top
ampTop = 0.02
freqTop = 1
freqSinTop = 3
freqCosTop = 8
offsetTop = 0.4*Lx
# bot
ampBot = ampTop
freqBot = freqTop
freqSinBot = freqSinTop
freqCosBot = freqCosTop
offsetBot = offsetTop+0.5*Lx
# top
y = y + y/Ly * ampTop * sin(2*pi*freqTop*(x-offsetTop)/Lx+freqSinTop*sin(2*pi*freqTop*(x-offsetTop)/Lx)+freqCosTop*cos(2*pi*freqTop*(x-offsetTop)/Lx))
# bot
y = y + (1-y/Ly) * ampBot * sin(2*pi*freqBot*(x-offsetBot)/Lx+freqSinBot*sin(2*pi*freqBot*(x-offsetBot)/Lx)+freqCosBot*cos(2*pi*freqBot*(x-offsetBot)/Lx))
f = Function(Vc).interpolate(as_vector([x, y]))
mesh.coordinates.assign(f)

#mesh = Mesh('mesh.msh')


nPerCore = abs(sqrt(mesh.num_entities(2)/(4)))  # seems to work but not super accurate
utils.print("sqrt(n^2 / core) ", nPerCore)

utils.writeInfoFile()


n = FacetNormal(mesh)
tau = as_vector((-n[1],n[0]))

if uSpace == "Hdiv":
	V_u = FunctionSpace(mesh, "RT", order+1)
elif uSpace == "Lag":
	V_u = VectorFunctionSpace(mesh, "CG", order+1)
V_p = FunctionSpace(mesh, "CG", order)
V_t = FunctionSpace(mesh, "CG", order)


#alpha = Function(V_p,name="alpha").interpolate(conditional(x<Lx/2.0, 0.001, 1000.0))
alpha = 1.0
#alphaFile = File(dataFolder+"alpha.pvd", comm = comm).write(alpha)

Z = V_u * V_p * V_t

upt = Function(Z)
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
		#- (inner(dot(n,grad(theta)),s)+inner(dot(n,grad(thetaOld)),s))*ds term?!?!?!?!?!?!?!
	)
	+ inner(u,grad(q))*dx
)

#https://gmd.copernicus.org/preprints/gmd-2021-367/gmd-2021-367.pdf


F_crankNicolson_freeFall_NavSlip_hDiv = (
	inner(u-uOld,v)*dx
	+ dt*(
		1.0/2.0*inner(dot(u, nabla_grad(u)) + dot(uOld, nabla_grad(uOld)), v)*dx
		+ sqrt(Pr/Ra) * nu * inner(Du + DuOld, Dv)*dx
		+ sqrt(Pr/Ra) * nu * inner(u + uOld,alpha * v)*ds
		- 1.0/2.0*inner(theta+thetaOld,v[1])*dx
	)
	+ inner(theta-thetaOld,s)*dx
	+ dt*( 
		1.0/2.0*inner(dot(u,grad(theta))+dot(uOld,grad(thetaOld)),s)*dx
		+ 1.0/(2.0*sqrt(Pr*Ra)) * kappa * inner(grad(theta+thetaOld), grad(s))*dx
		#- inner(dot(n,grad(theta+thetaOld)),s)*ds term?!?!?!?!?!?!?!
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
		+ 1.0/(1.0*sqrt(Pr*Ra)) * kappa * (inner(grad(theta), grad(s)))*dx
		#- inner(dot(n,grad(theta)),s)*ds term?!?!?!?!?!?!?!
	)
	+ inner(u,grad(q))*dx
	+ inner(grad(p),v)*dx
)


F = F_crankNicolson_freeFall_NavSlip_hDiv


# initial conditions for u
u = project(Constant([0.0,0.0]), V_u)
uOld.assign(u)

x, y = SpatialCoordinate(mesh)
#icTheta = 0.5*sin(2*pi*x/Lx)*exp(-4*(0.5-y)**2)+0.5


#icTheta = 10.0*sin(2*pi*x/Lx)*sin(2*pi*y/Ly)*exp(-1*(x**2+y**2))
expFactor = 0.5
[amp, x0, y0] = [10.0, -1.0, 2.0]
#icTheta = amp*exp(-expFactor*((x-x0)**2+(y-y0)**2))
[amp, x0, y0] = [-10.0, 2.0, 2.0]
#icTheta = icTheta + amp*exp(-expFactor*((x-x0)**2+(y-y0)**2))
[amp, x0, y0] = [-10.0, -3.0, 2.0]
#icTheta = icTheta + amp*exp(-expFactor*((x-x0)**2+(y-y0)**2))
[amp, x0, y0] = [10.0, 1.0, 1.0]
#icTheta = icTheta + amp*exp(-expFactor*((x-x0)**2+(y-y0)**2))
[amp, x0, y0] = [10.0, 6.0, 3.0]
#icTheta = icTheta + amp*exp(-expFactor*((x-x0)**2+(y-y0)**2))
[amp, x0, y0] = [-10.0, 4.0, 0.0]
#icTheta = icTheta + amp*exp(-expFactor*((x-x0)**2+(y-y0)**2))

icTheta = Constant(0.5)
theta = project(icTheta, V_t)

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
	#bcs.append(bcs_noSlip)
	

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
lastWrittenOutput = -1

def projectAvgFree(f, fOutput):
	avgF = 1/(Lx*Ly)*assemble(f*dx)
	fOutput.assign(f-avgF)
	return fOutput


def writeMeshFunctions():
	global lastWrittenOutput
	
	if nxOut != nx or nyOut != ny:
		thetaOut = project(theta, V_ptOut)
	else:
#		thetaOut = theta
		thetaOut = project(theta, V_ptOut)		
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
	lastWrittenOutput = t


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


uOld2 = u
thetaOld2 = theta

tWorld = datetime.datetime.now()

convergencewarningnumber = 0
revertSolverAfterXsolves = -1

def calcBdryL2(func):
	return assemble(inner(func,func)*ds)
	
	
def writeFactorError(fac, base):
		error = calcBdryL2(fac*alpha*dot(u,tau)+dot(dot(n,Du),tau))
		#utils.print("temp\t",fac,"\t",error, "\t", (error/base))
		utils.print("temp\t",fac,"\t", (error/base))

solverParams = 0		# (my) fast ones 0, firedrake default 1

while(t<tEnd):
	try:
		utils.print(solver.parameters)
		solver.solve()
	except ConvergenceError as convError:
		utils.print(convError)
		if "DIVERGED_LINEAR_SOLVE" in str(convError):
			if solverParams == 1:
				raise Exception("Diverged with both solverParams") # Don't! If you catch, likely to hide bugs.
			solverParams = 1
			revertSolverAfterXsolves = 1
			utils.print("DIVERGED_LINEAR_SOLVE, trying again with different solver parameters")
			solver = NonlinearVariationalSolver(problem, nullspace = nullspace)
			# there is an error where the linear solve doesn't converge. The reason seems to be that the 0 KSP preconditioned resid norm for the first try to linear solve is high (~ 80* the one of the usual first prec resid norm)
			# "solution" for now change the data a bit and try again		

			# TRY ANOTHER PRECONDITIONER AT ERROR!!!
			# CHECK IF IT IS THE RIGHT ERROR FIRST
			# PRINT ERROR STACK ANYWAYS
#			u.assign(1.0/11.0*(10*uOld + uOld2))
#			theta.assign(1.0/11.0*(10*thetaOld + thetaOld2))
			convergencewarningnumber += 1
			utils.putInfoInInfoString("DIVERGED_LINEAR_SOLVE WARNING "+str(convergencewarningnumber), "at time " + str(t + dt) + ": trying again different solver parameters")
			utils.writeInfoFile()
		else:
			utils.print("error in solve")
			raise Exception("convergence error")
			
	else:
			
		t = t + dt
		utils.setSimulationTime(t)
		u, p, theta = upt.subfunctions	# depending on the firedrake version have to either use upt.split() (old) or upt.subfunctions (newer)
		
		uOld2.assign(uOld)
		thetaOld2.assign(thetaOld)
		
		uOld.assign(u)
		thetaOld.assign(theta)
		
		if round(t,12) >= round(lastWrittenOutput + writeOutputEvery,12):
			writeMeshFunctions()
			
		utils.print(round(t/tEnd*100,9),"% done (after",datetime.datetime.now()-tWorld,"), t=",round(t,9))
		utils.print(" ")
		utils.print("div(u)\t",norm(div(u)))	
		utils.print(" ")
		utils.print("u n\t\t",calcBdryL2(dot(u,n)))
		utils.print("u tau\t",calcBdryL2(dot(u,tau)))	
		utils.print("n Du tau\t",calcBdryL2(dot(n,dot(tau,Du))))
		
#		utils.print("temp\t",factor,"\t",assemble(inner(factor*alpha*dot(u,tau)+dot(dot(n,Du),tau),factor*alpha*dot(u,tau)+dot(dot(n,Du),tau))*ds))
		utils.print(" ")
		base = 	calcBdryL2(dot(u,tau)) + calcBdryL2(dot(n,dot(tau,Du)))
#		factor = 0.25
		writeFactorError(0.25,base)
		writeFactorError(0.50,base)
		writeFactorError(0.75,base)
		writeFactorError(0.90,base)
		writeFactorError(1.00,base)
		writeFactorError(1.10,base)
		writeFactorError(1.25,base)
		writeFactorError(1.50,base)
		writeFactorError(1.75,base)
		writeFactorError(2.00,base)
		
		# seems to converge to the nav slip for order to infty
		# it should be the true solution if it solves the variational problem
		# at least up to comp errors
		# i guess/hope the var prob is more accurate then the derivative on the boundary
		
		
		utils.print(" ")
		utils.print(" ")
		tWorld = datetime.datetime.now()
		
		
		if solverParams == 1:
			utils.print("reverting to my parameters")
			solverParams = 0
			solver = NonlinearVariationalSolver(problem, nullspace = nullspace, solver_parameters=parameters_my, appctx=appctx)
	
utils.writeEndInfo()
