import os
os.environ["OMP_NUM_THREADS"] = "1"

import myUtilities
#from myMeshes import *
from firedrake import *
import datetime
import weakref

outputFolder = "output/"

my_ensemble = Ensemble(COMM_WORLD, COMM_WORLD.size)
comm = my_ensemble.comm

utils = myUtilities.utils(comm, outputFolder)
utils.generateRecoveryScript(__file__)



### PARAMETERS ###
nXY = 256

nOut = nXY

uSpace = "Lag"			# either Hdiv or Lag

dt = 0.03
writeOutputEveryXsteps = 1
writeOutputEvery = dt*writeOutputEveryXsteps			# write mesh functions
writeUP = True						# output u and p?

Lx = 2.0
Ly = 1.0

t = 0.0
tEnd = 100 #1.0

### only nav slip ###
alpha = 100000.0		# alpha in tau Du n = alpha tau u

				
nu = 1.0			# ... - nu * Laplace u ...
kappa = 0.0001			# ... - kappa * Laplace theta ...
Ra = 1.0			# ... + Ra * theta * e_2
Pr = 1.0			# 1/Pr*(u_t+u cdot nabla u) + ...

#ampFreqOffsetList = [[0.15,1.0,0]] # examples: [[0.1,1.0,0.0],[0.1,2.0,0.0]] or [[0.1,1.0,pi/2.0]]
meshDiagonal = "crossed"			# "left"= /	"right"= \	crossed = "X"

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
nPerCore = abs(sqrt(nx*ny/COMM_WORLD.size))
utils.print("n / core ", nPerCore)

utils.putInfoInInfoString("nXY",nXY)
utils.putInfoInInfoString("n/core",nPerCore)
utils.putInfoInInfoString("dt",dt)
utils.putInfoInInfoString("Lx",Lx)
utils.putInfoInInfoString("Ly",Ly)
utils.putInfoInInfoString("tEnd",tEnd)
utils.putInfoInInfoString("alpha",alpha)
utils.putInfoInInfoString("kappa",kappa)
utils.putInfoInInfoString("Ra",Ra)
utils.putInfoInInfoString("projectPoutputToAverageFree",projectPoutputToAverageFree)
utils.putInfoInInfoString("dataFolder",dataFolder)



### mesh ###
mesh = Mesh('mesh.msh')
#mesh = PeriodicRectangleMesh(nx,ny,Lx,Ly, "x", comm = comm, diagonal = meshDiagonal)	# mesh Lx=Gamma in e_1, Ly in e_2, periodic in x=e_1 dir
#boundary_id_bot = 1
#boundary_id_top = 2




utils.writeInfoFile()



n = FacetNormal(mesh)
tau = as_vector((-n[1],n[0]))
x,y = SpatialCoordinate(mesh)




order = 1
if uSpace == "Hdiv":
	V_u = FunctionSpace(mesh, "RT", order+1)
elif uSpace == "Lag":
	V_u = VectorFunctionSpace(mesh, "CG", order+1)
V_p = FunctionSpace(mesh, "CG", order)
V_t = FunctionSpace(mesh, "CG", order)


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
Dv = 0.5*(grad(v)+nabla_grad(v))
v_n = dot(n,v)*n
v_tau = dot(tau,v)*tau
u_n = dot(n,u)*n
u_tau = dot(tau,u)*tau

#F_bsqBackward_navSlip = (
#	inner(u-uOld,v)*dx
#	+ dt*(
#		inner(dot(u, nabla_grad(u)), v)*dx
#		- inner(p,div(v))*dx
#		+ 2.0 * inner(Du, Dv)*dx
#		- Ra * inner(theta,v[1])*dx
#		+ 2.0 * alpha * inner(u,v_tau)*ds
#	)
#	+ inner(theta-thetaOld,s)*dx
#	+ dt*( 
#		kappa * inner(grad(theta), grad(s))*dx
#		+ inner(dot(u,grad(theta)),s)*dx
#	)
#	+ inner(div(u),q)*dx
#)

F_rb_backwards = (
	inner(u-uOld,v)*dx
	+ dt*(
		inner(dot(u,nabla_grad(u)),v)*dx
		- inner(p,div(v))*dx
		+ nu * 2.0 * inner(Du, grad(v))*dx
		- Ra * inner(theta,v[1])*dx
		+ alpha * nu * 2.0 * inner(u,v)*ds
	)
	
	+ inner(theta-thetaOld,s)*dx
	+ dt*(
		inner(dot(u,grad(theta)),s)*dx
		+ kappa * inner(grad(theta),grad(s))*dx
#?!?!?!!?!	#- kappa * inner(dot(n,grad(theta)),s)*ds

	)
	
	+ inner(u,grad(q))*dx
)

F_rb_backwards_noSlip = (
	inner(u-uOld,v)*dx
	+ dt*(
		inner(dot(u,nabla_grad(u)),v)*dx
		+ inner(grad(p),v)*dx
		+ nu * inner(grad(u), grad(v))*dx
		- Ra * inner(theta,v[1])*dx
	)
	
	+ inner(theta-thetaOld,s)*dx
	+ dt*(
		inner(dot(u,grad(theta)),s)*dx
		+ kappa * inner(grad(theta),grad(s))*dx
#?!?!?!!?!	#- kappa * inner(dot(n,grad(theta)),s)*ds

	)
	
	+ inner(u,grad(q))*dx
)


F_rb_crankNicolson_noSlip = (
	1.0/Pr * inner(u-uOld,v)*dx
	+ dt*(
		1.0/Pr * 1.0/2.0*(inner(dot(u, nabla_grad(u)), v)+inner(dot(uOld, nabla_grad(uOld)), v))*dx
		+ inner(grad(p),v)*dx
		+ nu * 1.0/2.0*(inner(grad(u), grad(v))+inner(grad(uOld), grad(v)))*dx
		- Ra * 1.0/2.0*(inner(theta,v[1]) + inner(thetaOld,v[1]))*dx
	)
	+ inner(theta-thetaOld,s)*dx
	+ dt*( 
		1.0/2.0*(inner(dot(u,grad(theta)),s)+inner(dot(uOld,grad(thetaOld)),s))*dx
		+ kappa * 1.0/2.0*(inner(grad(theta), grad(s))+inner(grad(thetaOld), grad(s)))*dx
#?!?!?!!?!	#- kappa * 1.0/2.0*(inner(dot(n,grad(theta)),s)+inner(dot(n,grad(thetaOld)),s))*ds
	)
	+ inner(u,grad(q))*dx
)



F = F_rb_crankNicolson_noSlip


# initial conditions for u
u = project(Constant([0.0,0.0]), V_u)
uOld.assign(u)


x, y = SpatialCoordinate(mesh)
#icTheta = 10.0*sin(2*pi*x/Lx)*sin(2*pi*y/Ly)*exp(-1*(x**2+y**2))
[amp, x0, y0] = [10.0, -1.0, -2.0]
icTheta = amp*exp(-1*((x-x0)**2+(y-y0)**2))
[amp, x0, y0] = [-10.0, 4.0, -2.0]
icTheta = icTheta + amp*exp(-1*((x-x0)**2+(y-y0)**2))
[amp, x0, y0] = [-10.0, -3.0, -2.0]
icTheta = icTheta + amp*exp(-1*((x-x0)**2+(y-y0)**2))
[amp, x0, y0] = [10.0, 1.0, -1.0]
icTheta = icTheta + amp*exp(-1*((x-x0)**2+(y-y0)**2))
[amp, x0, y0] = [10.0, 6.0, -3.0]
icTheta = icTheta + amp*exp(-1*((x-x0)**2+(y-y0)**2))
[amp, x0, y0] = [-10.0, 4.0, 0.0]
icTheta = icTheta + amp*exp(-1*((x-x0)**2+(y-y0)**2))
#icTheta = Constant(0.)
theta = project(icTheta, V_t)
thetaOld.assign(theta)

bcs = []

#bc_rbBot = DirichletBC(Z.sub(2), Constant(1.0), (boundary_id_bot))
#bc_rbTop = DirichletBC(Z.sub(2), Constant(0.0), (boundary_id_top))
#bcs.append(bc_rbBot)
#bcs.append(bc_rbTop)


u_n = dot(u,n)
v_n = dot(v,n)




#bc_u = EquationBC(inner(dot(n,Du) - alpha*u,v_tau)*ds+inner(u,v_n)*ds==0, u, (1,2), V=Z.sub(0))
#bc_uN = EquationBC(inner(dot(u,n),dot(v,n))*ds==0, u, (1,2), V=Z.sub(0))
#bc_uTau = EquationBC(inner(u,v_tau)*ds==0, u, (1,2), V=Z.sub(0))
#bc_uNav = EquationBC(inner(dot(Du,n)-alpha*u,v)*ds==0, u, (1,2), V=Z.sub(0), bcs=bc_uN)
#navSlipBc = EquationBC(inner(dot(n,Du)+alpha*u,v_tau)*ds+inner(u,v_n)*ds==0, u, "on_boundary", V=Z.sub(0))


#bc_t1 = EquationBC(inner(u+0.00005*dot(n,Du),v_tau)*ds+inner(u,v_n)*ds==0, u, (1,2), V=Z.sub(0))
#bc_un0 = EquationBC(inner(dot(u,n),dot(v,n))*ds==0, u, (1,2), V=Z.sub(0))
#bc_t3 = EquationBC(inner(dot(dot(grad(u)+nabla_grad(u),n)+alpha*u,tau),dot(v,tau))*ds==0, u, (1,2), V=Z.sub(0), bcs=bc_un0)
#bc_t4 = EquationBC(inner(u,v)*ds==0, u, (1,2), V=Z.sub(0))
#bc_utau0 = EquationBC(inner(u,v_tau)*ds==0, u, (1,2), V=Z.sub(0))
#bc_t6 = EquationBC(inner(u,v_tau)*ds+inner(u,v_n)*ds==0, u, (1,2), V=Z.sub(0))

#if uSpace == "Hdiv":
	#bc_noPenHdiv = DirichletBC(Z.sub(0), Constant((0.0,0.0)), (boundary_id_bot,boundary_id_top)) # because of the hdiv space setting 0 bc for it is only setting u cdot n = 0 #https://github.com/firedrakeproject/firedrake/issues/169#issuecomment-34557942
	#bcs.append(bc_noPenHdiv)
#elif uSpace == "Lag":
	#bcs_noSlip = DirichletBC(Z.sub(0), Constant((0.0,0.0)), (boundary_id_bot,boundary_id_top))
	#bcs.append(bcs_noSlip)
	#bc_uN = EquationBC(inner(dot(u,n),dot(v,n))*ds==0, u, (boundary_id_bot,boundary_id_top), V=Z.sub(0))
	#bc_uN_nav = EquationBC(inner(dot(tau,dot(Du,n))+alpha*dot(u,tau),dot(v,tau))*ds==0, u, (boundary_id_bot,boundary_id_top), V=Z.sub(0), bcs=bc_uN)
	#bc_uN_nav = EquationBC(inner(dot(tau,dot(Du,n))+alpha*dot(u,tau),dot(v,tau))*ds+inner(dot(u,n),dot(v,n))*ds==0, u, (boundary_id_bot,boundary_id_top), V=Z.sub(0))
	#bcs.append(bc_uN_nav)
	

	

problem = NonlinearVariationalProblem(F, upt, bcs = bcs)


nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True), Z.sub(2)])


parameters_easySplit = {	# splits the solving of the nse and the temperature equation and nothing else
	# seems to be more robust but is super slow on big grids
	"pc_type": "fieldsplit",
	"pc_fieldsplit_type": "multiplicative",		# additive multiplicative symmetric_multiplicative special schur gkb
	"pc_fieldsplit_0_fields": "0,1",
	"pc_fieldsplit_1_fields": "2",
}


#solver = NonlinearVariationalSolver(problem, nullspace = nullspace, solver_parameters=parameters_easySplit)
solver = NonlinearVariationalSolver(problem, nullspace = nullspace)

uptFile = File(dataFolder+"upt.pvd", comm = COMM_WORLD)
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





tWorld = datetime.datetime.now()
while(t<tEnd):
	
	solver.solve()
	
	
	t = t + dt
	utils.setSimulationTime(t)
	u, p, theta = upt.split()
	uOld.assign(u)
	thetaOld.assign(theta)
	
	if round(t,12) >= round(lastWrittenOutput + writeOutputEvery,12):
		writeMeshFunctions()
	#utils.print("u\t",assemble(inner(u,u)*ds))	
	#utils.print("u tau\t",assemble(inner(u,tau)*ds))	
	#utils.print("u n\t",assemble(inner(u,n)*ds))	
	#utils.print("n Du tau\t",assemble(inner(dot(n,Du),tau)*ds))
	#utils.print("temp\t",assemble(inner(alpha*u+dot(n,Du),tau)*ds))
	
	
	utils.print(round(t/tEnd*100,12),"% done (after",datetime.datetime.now()-tWorld,"), t=",round(t,9))
	tWorld = datetime.datetime.now()
utils.writeEndInfo()


