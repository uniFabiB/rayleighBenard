import os
os.environ["OMP_NUM_THREADS"] = "1"

import myUtilities
from myMeshes import *
from firedrake import *
import datetime
import weakref

outputFolder = "output/"

my_ensemble = Ensemble(COMM_WORLD, COMM_WORLD.size)
comm = my_ensemble.comm

utils = myUtilities.utils(comm, outputFolder)
utils.generateRecoveryScript(__file__)



### PARAMETERS ###
n = 512

nOut = n

tau = 0.00001
writeOutputEveryXsteps = 20
writeOutputEvery = tau*writeOutputEveryXsteps			# write mesh functions
writeUP = True						# output u and p?

Lx = 1.0
Ly = 1.0

t = 0.0
tEnd = 1.0

### only nav slip ###
Ls = 0.01			# alpha = 2/L_s in navier slip as Du = 1/2 partial_1 u_2 on top and bottom

				
nu = 1.0			# ... - nu * Laplace u ...
kappa = 1.0			# ... - kappa * Laplace theta ...
Ra = 10.0**8			# ... + Ra * theta * e_2
Pr = 1.0			# 1/Pr*(u_t+u cdot nabla u) + ...

meshTyp = "myMeshHigherOrder"				#"periodicRectangle", "myMesh", "myMeshHigherOrder"
meshDiagonal = "crossed"			# "left"= /	"right"= \	crossed = "X"

slip = "noSlip"		# "noSlip"	"navSlip"

problem = "rayleighBenard"		# "boussinesq"	"rayleighBenard"
### PARAMETERS END ###


### OPTIONS ###
projectPoutputToAverageFree = True 	# force the output function of p to be average free (doesn't change the calculation)

dataFolder = outputFolder + "data/"


nx = round(n*Lx/Ly)
ny = n

nxOut = round(nOut*Lx/Ly)
nyOut = nOut

printErrors = True
### OPTIONS END ###


#nCells = nx*ny*2	for diagonal "left" or "right"
#nCells = nx*ny*4	for diagonal "crossed"
utils.print("n / core ", abs(sqrt(nx*ny/COMM_WORLD.size)))

utils.putInfoInInfoString("n",n)
utils.putInfoInInfoString("tau",tau)
utils.putInfoInInfoString("Lx",Lx)
utils.putInfoInInfoString("Ly",Ly)
utils.putInfoInInfoString("tEnd",tEnd)
utils.putInfoInInfoString("Ls",Ls)
utils.putInfoInInfoString("kappa",kappa)
utils.putInfoInInfoString("Ra",Ra)
utils.putInfoInInfoString("problem",problem)
utils.putInfoInInfoString("meshTyp",meshTyp)
utils.putInfoInInfoString("projectPoutputToAverageFree",projectPoutputToAverageFree)
utils.putInfoInInfoString("dataFolder",dataFolder)




if meshTyp == "periodicRectangle":
	mesh = PeriodicRectangleMesh(nx,ny,Lx,Ly, "x")	# mesh Lx=Gamma in e_1, Ly in e_2, periodic in x=e_1 dir
	boundary_id_bot = 1
	boundary_id_top = 2
elif meshTyp == "myMesh":
	meshAmpl = 0.02
	meshFreq = 6.0
	meshPhi0 = 0.0
	mesh = myPartiallyPeriodicRectangleMesh(nx, ny, Lx, Ly, meshAmpl, meshFreq, meshPhi0, comm = comm, diagonal = meshDiagonal)
	meshOut = myPartiallyPeriodicRectangleMesh(nxOut, nyOut, Lx, Ly, meshAmpl, meshFreq, meshPhi0, comm = comm, diagonal = meshDiagonal)
	#kappa = gamma''/(1+gamma'^2)^(3/2)		#https://de.wikipedia.org/wiki/Kr%C3%BCmmung#Ebene_Kurven
	# gamma(x_1)= A sin(2pi freq x_1/Gamma)
	# -> kappa(x_1)=-A (2 pi freq/Gamma)^2 sin(2 pi freq/Gamma x_1) / (1+ ( A 2 pi freq/Gamma)^2 cos^2(2pi freq x_1/Gamma)) 
	# max |kappa| = A (2 pi freq/Gamma)^2 = meshAmpl *(2.0 * pi * meshFreq/Lx)**2
	maxKappa = meshAmpl *(2.0 * pi * meshFreq/Lx)**2
	utils.print("maxKappa",maxKappa)
	utils.putInfoInInfoString("maxKappa",maxKappa)
	boundary_id_bot = 1
	boundary_id_top = 2
	boundary_ids = (boundary_id_bot, boundary_id_top)
elif meshTyp == "myMeshHigherOrder":
	meshAmplBotList = [0.01, 0.01, 0.01]
	meshFreqBotList = [8.0, 2.0, 1.0]
	meshPhi0BotList = [0.0, 0.0, 0.0]
	meshAmplTopList = meshAmplBotList
	meshFreqTopList = meshFreqBotList
	meshPhi0TopList = meshPhi0BotList
	mesh = myHigherOrderNonSymmetricPartiallyPeriodicRectangleMesh(nx, ny, Lx, Ly, meshAmplBotList, meshFreqBotList, meshPhi0BotList, meshAmplTopList, meshFreqTopList, meshPhi0TopList, comm = comm, diagonal = meshDiagonal)
	meshOut = myHigherOrderNonSymmetricPartiallyPeriodicRectangleMesh(nxOut, nyOut, Lx, Ly, meshAmplBotList, meshFreqBotList, meshPhi0BotList, meshAmplTopList, meshFreqTopList, meshPhi0TopList, comm = comm, diagonal = meshDiagonal)
	
	#maxKappa = max(meshAmplBot *(2.0 * pi * meshFreqBot/Lx)**2, meshAmplTop *(2.0 * pi * meshFreqTop/Lx)**2)
	boundary_id_bot = 1
	boundary_id_top = 2
	boundary_ids = (boundary_id_bot, boundary_id_top)
else:
        raise Error("ERROR mesh not specified")
        




utils.writeInfoFile()



normal = FacetNormal(mesh)
tangential = as_vector((-normal[1],normal[0]))
x,y = SpatialCoordinate(mesh)




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




nu = Constant(float(nu))
Ls = Constant(float(Ls))
kappa = Constant(float(kappa))
Ra = Constant(float(Ra))
Pr = Constant(float(Pr))



F_crankNicolson_navSlip = (
	1.0/Pr * inner(u-uOld,v)*dx
	+ tau*(
		 1.0/Pr * 1.0/2.0*(inner(dot(u, nabla_grad(u)), v)+inner(dot(uOld, nabla_grad(uOld)), v))*dx
		+ inner(grad(p),v)*dx
		+ nu * 1.0/2.0*(inner(grad(u), grad(v))+inner(grad(uOld), grad(v)))*dx
		+ nu * 1.0/Ls * 1.0/2.0*(inner(u[0], v[0])+inner(uOld[0], v[0]))*ds
		- Ra * 1.0/2.0*(inner(theta,v[1]) + inner(thetaOld,v[1]))*dx
	)
	+ inner(theta-thetaOld,s)*dx
	+ tau*( 
		kappa * 1.0/2.0*(inner(grad(theta), grad(s))+inner(grad(thetaOld), grad(s)))*dx
		+ 1.0/2.0*(inner(dot(u,grad(theta)),s)+inner(dot(uOld,grad(thetaOld)),s))*dx
	)
	+ inner(u,grad(q))*dx
)


F_crankNicolson_noSlip = (
	1.0/Pr * inner(u-uOld,v)*dx
	+ tau*(
		 1.0/Pr * 1.0/2.0*(inner(dot(u, nabla_grad(u)), v)+inner(dot(uOld, nabla_grad(uOld)), v))*dx
		+ inner(grad(p),v)*dx
		+ nu * 1.0/2.0*(inner(grad(u), grad(v))+inner(grad(uOld), grad(v)))*dx
		- Ra * 1.0/2.0*(inner(theta,v[1]) + inner(thetaOld,v[1]))*dx
	)
	+ inner(theta-thetaOld,s)*dx
	+ tau*( 
		kappa * 1.0/2.0*(inner(grad(theta), grad(s))+inner(grad(thetaOld), grad(s)))*dx
		+ 1.0/2.0*(inner(dot(u,grad(theta)),s)+inner(dot(uOld,grad(thetaOld)),s))*dx
	)
	+ inner(u,grad(q))*dx
)

F_bsqBackward_navSlip = (
	1.0/Pr * inner(u-uOld,v)*dx
	+ tau*(
		 1.0/Pr * inner(dot(u, nabla_grad(u)), v)*dx
		+ inner(grad(p),v)*dx
		+ nu * inner(grad(u), grad(v))*dx
		+ nu * 1.0/Ls * inner(u[0], v[0])*ds
		- Ra * inner(theta,v[1])*dx
	)
	+ inner(theta-thetaOld,s)*dx
	+ tau*( 
		kappa * inner(grad(theta), grad(s))*dx
		+ inner(dot(u,grad(theta)),s)*dx
	)
	+ inner(u,grad(q))*dx
)

F_bsqBackward_noSlip = (
	1.0/Pr * inner(u-uOld,v)*dx
	+ tau*(
		 1.0/Pr * inner(dot(u, nabla_grad(u)), v)*dx
		+ inner(grad(p),v)*dx
		+ nu * inner(grad(u), grad(v))*dx
		- Ra * inner(theta,v[1])*dx
	)
	+ inner(theta-thetaOld,s)*dx
	+ tau*( 
		kappa * inner(grad(theta), grad(s))*dx
		+ inner(dot(u,grad(theta)),s)*dx
	)
	+ inner(u,grad(q))*dx
)

if problem == "boussinesq":
	if slip == "noSlip":
		scheme = "backward euler noSlip"
		F = F_bsqBackward_noSlip
	elif slip == "navSlip":
		scheme = "backward euler navSlip"
		F = F_bsqBackward_navSlip
	else:
		utils.warn("slip (",slip,") is not correct")
elif problem == "rayleighBenard":
	if slip == "noSlip":
		scheme = "crank nicolson noSlip"
		F = F_crankNicolson_noSlip
	elif slip == "navSlip":
		scheme = "crank nicolson navSlip"
		F = F_crankNicolson_navSlip
	else:
		utils.warn("slip (",slip,") is not correct")
else:
	utils.warn("problem not correct specified")

if slip == "navSlip" and meshTyp == "myMesh":
	utils.warn("navSlip does not work with curved mesh!!!")
	
	
utils.putInfoInInfoString("scheme",scheme)

# initial conditions for u
u = project(Constant([0.0,0.0]), V_u)
uOld.assign(u)


if problem == "boussinesq":
	icTheta = 1.0*sin(pi*y/Ly+2*pi*x/Lx)*sin(2*pi*x/Lx+2*pi*y/Ly)
	#icTheta = 1.0*exp(-(1.0*(y-0.5*Ly)**2))*sin(2*pi*x/Lx)
	theta = project(icTheta, V_t)
elif problem == "rayleighBenard":
	icTheta1 = 0.001*exp(-(10.0*(y-0.5*Ly)**2))*sin(2*pi*x/Lx)
	icTheta2 = 0.0001*exp(-(10.0*(y-0.5*Ly)**2))*exp(-(10.0*(x/2.0-0.5*Lx)**2))
	theta = project(icTheta1+icTheta2, V_t)
#	theta = project(Constant(0.0), V_t)
else:
	utils.warn("problem not correct specified") 

thetaOld.assign(theta)

bcs = []

if problem == "rayleighBenard":
	bc_rbBot = DirichletBC(Z.sub(2), Constant(1.0), (boundary_id_bot))
	bc_rbTop = DirichletBC(Z.sub(2), Constant(-1.0), (boundary_id_top))
	bcs.append(bc_rbBot)
	bcs.append(bc_rbTop)

if slip == "noSlip":
	bc_noSlip = DirichletBC(Z.sub(0), Constant([0.0, 0.0]), "on_boundary")
	bcs.append(bc_noSlip)
elif slip == "navSlip":
	bc_noPen = DirichletBC(Z.sub(0).sub(1), Constant(0.0), "on_boundary")
	bcs.append(bc_noPen)
	
	

problem = NonlinearVariationalProblem(F, upt, bcs = bcs)


nullspace = MixedVectorSpaceBasis(Z, [Z.sub(0), VectorSpaceBasis(constant=True), Z.sub(2)])
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
		thetaOut = theta
		
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
	t = t + tau
	utils.setSimulationTime(t)
	u, p, theta = upt.split()
	uOld.assign(u)
	thetaOld.assign(theta)
	
	if round(t,12) >= round(lastWrittenOutput + writeOutputEvery,12):
		writeMeshFunctions()
	
	utils.print(round(t/tEnd*100,12),"% done (after",datetime.datetime.now()-tWorld,"), t=",round(t,12))
	if printErrors:
		utils.print("\tl2 div(u) / l2 grad(u)", round(norm(div(u))/norm(grad(u)),12))
		expression1 = grad(u)[0,1]+1.0/Ls*u[0]
		expression2 = grad(u)[0,1]-1.0/Ls*u[0]
		bdryError1 = abs(sqrt(assemble(inner(expression1,expression1)*ds(boundary_id_top))))
		bdryError2 = abs(sqrt(assemble(inner(expression2,expression2)*ds(boundary_id_bot))))
		bdryError = bdryError1+bdryError2
		utils.print("\tl2 partial_2 u_1 + 1/Ls u_1", bdryError)			#grad(v)[i,j] = v[i].dx(j)		nabla_grad(v)[i,j] = v[j].dx(i)
		LsU1 = abs(sqrt(assemble(inner(1.0/Ls*u[0],1.0/Ls*u[0])*ds)))
		utils.print("\tl2 u_1", LsU1)
		partial2U1 = abs(sqrt(assemble(inner(grad(u)[0,1],grad(u)[0,1])*ds)))
		utils.print("\tl2 partial_2 u_1", partial2U1)
		if LsU1+partial2U1 != 0:
			utils.print("\tls Du + au / (l2 u1 + l2 partial_2 u_1)", round(bdryError/(LsU1+partial2U1),12))
	tWorld = datetime.datetime.now()
utils.writeEndInfo()


