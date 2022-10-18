### mesh with sinusoidal boundary on top and bottom

# https://firedrakeproject.org/_modules/firedrake/utility_meshes.html
# utility_meshes imports
import numpy as np

import ufl

from pyop2.mpi import COMM_WORLD
from firedrake.utils import IntType, RealType, ScalarType

from firedrake import VectorFunctionSpace, Function, Constant, \
    par_loop, dx, WRITE, READ, interpolate, FiniteElement, interval, tetrahedron
from firedrake.cython import dmcommon
from firedrake import mesh
from firedrake import function
from firedrake import functionspace
from firedrake.petsc import PETSc




# my imports
from firedrake import CylinderMesh
from firedrake import SpatialCoordinate
from firedrake import atan_2
from firedrake import as_vector
from firedrake import sin
from warnings import warn


def myPartiallyPeriodicRectangleMesh(nx, ny, Lx, Ly, meshAmplitude, meshFrequency, meshPhiOffset, quadrilateral=False, reorder=None, distribution_parameters=None, diagonal=None, comm=COMM_WORLD, name=mesh.DEFAULT_MESH_NAME):
    return myHigherOrderNonSymmetricPartiallyPeriodicRectangleMesh(nx, ny, Lx, Ly, [meshAmplitude], [meshFrequency], [meshPhiOffset], [meshAmplitude], [meshFrequency], [meshPhiOffset], quadrilateral=False, reorder=None, distribution_parameters=None, diagonal=None, comm=COMM_WORLD, name=mesh.DEFAULT_MESH_NAME)

    
def myHigherOrderNonSymmetricPartiallyPeriodicRectangleMesh(nx, ny, Lx, Ly, meshAmplitudeBotList, meshFrequencyBotList, meshPhiOffsetBotList, meshAmplitudeTopList, meshFrequencyTopList, meshPhiOffsetTopList, quadrilateral=False, reorder=None, distribution_parameters=None, diagonal=None, comm=COMM_WORLD, name=mesh.DEFAULT_MESH_NAME):
    direction="x"
    ### from firedrake PartiallyPeriodicRectangleMesh
    
    """Generates RectangleMesh that is periodic in the x or y direction.

    :arg nx: The number of cells in the x direction
    :arg ny: The number of cells in the y direction
    :arg Lx: The extent in the x direction
    :arg Ly: The extent in the y direction
    :kwarg direction: The direction of the periodicity (default x).
    :kwarg quadrilateral: (optional), creates quadrilateral mesh, defaults to False
    :kwarg reorder: (optional), should the mesh be reordered
    :kwarg diagonal: (optional), one of ``"crossed"``, ``"left"``, ``"right"``. ``"left"`` is the default.
        Not valid for quad meshes.
    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).
    :kwarg name: Optional name of the mesh.

    If direction == "x" the boundary edges in this mesh are numbered as follows:

    * 1: plane y == 0
    * 2: plane y == Ly

    If direction == "y" the boundary edges are:

    * 1: plane x == 0
    * 2: plane x == Lx
    """

    if direction not in ("x", "y"):
        raise ValueError("Unsupported periodic direction '%s'" % direction)

    # handle x/y directions: na, La are for the periodic axis
    na, nb, La, Lb = nx, ny, Lx, Ly
    if direction == "y":
        na, nb, La, Lb = ny, nx, Ly, Lx

    if na < 3:
        raise ValueError("2D periodic meshes with fewer than 3 \
cells in each direction are not currently supported")

    m = CylinderMesh(na, nb, 1.0, 1.0, longitudinal_direction="z", quadrilateral=quadrilateral, reorder=reorder,distribution_parameters=distribution_parameters, diagonal=diagonal, comm=comm, name=name)
    ### end from firedrake PartiallyPeriodicRectangleMesh
    
    
    
    
    ### my changes
    
    # error handling if bot amp, freq, offset sizes don't match
    if len(meshAmplitudeBotList) != len(meshFrequencyBotList):
    	raise IndexError("meshAmplitudeBotList has to have the same size as meshFrequencyBotList as they build the sinusoidal boundary")
    if len(meshAmplitudeBotList) != len(meshPhiOffsetBotList):
    	# warning doesnt warn so raise error
    	raise IndexError("meshAmplitudeBotList has to have the same size as meshPhiOffsetBotList as they build the sinusoidal boundary")
    	#warn("meshPhiOffsetBotList.size does not equal meshAmplitudeBotList.size! additional phi offsets will be ignored! missing phi offsets will be set to 0")
    	# the following doesn't get executed
    	if len(meshAmplitudeBotList) > len(meshPhiOffsetBotList):
    		newOffsets = np.zeros(len(meshAmplitudeBotList))
    		for i in range(len(meshPhiOffsetBotList)):
    			newOffsets[i] = meshPhiOffsetBotList[i]
    		meshPhiOffsetBotList = newOffsets
		
    # error handling if top amp, freq, offset sizes don't match
    if len(meshAmplitudeTopList) != len(meshFrequencyTopList):
    	raise IndexError("meshAmplitudeBotList has to have the same size as meshFrequencyBotList as they build the sinusoidal boundary")
    if len(meshAmplitudeTopList) != len(meshPhiOffsetTopList):
    	raise IndexError("meshAmplitudeTopList has to have the same size as meshPhiOffsetTopList as they build the sinusoidal boundary")
    	#warn("meshPhiOffsetBotList.size does not equal meshAmplitudeBotList.size! additional phi offsets will be ignored! missing phi offsets will be set to 0")
    	# the following doesn't get executed
    	if len(meshAmplitudeTopList) > len(meshPhiOffsetTopList):
    		newOffsets = np.zeros(len(meshAmplitudeTopList))
    		for i in range(len(meshPhiOffsetTopList)):
    			newOffsets[i] = meshPhiOffsetTopList[i]
    		meshPhiOffsetTopList = newOffsets
		
    # get mesh coords
    meshx,meshy,meshz = SpatialCoordinate(m)
    meshphi = atan_2(meshx,meshy)
    newZ = meshz
    
    # change z coord
    # bot
    for i in range(len(meshAmplitudeBotList)):
    	newZ = newZ+newZ/Ly*meshAmplitudeBotList[i]/Lb*sin(meshFrequencyBotList[i]*(meshphi+meshPhiOffsetBotList[i]))
    # top
    for i in range(len(meshAmplitudeTopList)):
    	newZ = newZ+(1-meshz/Ly)*meshAmplitudeTopList[i]/Lb*sin(meshFrequencyTopList[i]*(meshphi+meshPhiOffsetTopList[i]))

    newCoords = Function(m.coordinates.function_space()).interpolate(as_vector([meshx, meshy, newZ]))
    m.coordinates.assign(newCoords)
    
    ### end my changes
    
    
    
    
    
    ### from firedrake PartiallyPeriodicRectangleMesh
    
    coord_family = 'DQ' if quadrilateral else 'DG'
    cell = 'quadrilateral' if quadrilateral else 'triangle'
    coord_fs = VectorFunctionSpace(m, FiniteElement(coord_family, cell, 1, variant="equispaced"), dim=2)
    old_coordinates = m.coordinates
    new_coordinates = Function(coord_fs, name=mesh._generate_default_mesh_coordinates_name(name))

    # make x-periodic mesh
    # unravel x coordinates like in periodic interval
    # set y coordinates to z coordinates
    domain = "{[i, j, k, l]: 0 <= i, k < old_coords.dofs and 0 <= j < new_coords.dofs and 0 <= l < 3}"
    instructions = f"""
    <{RealType}> Y = 0
    <{RealType}> pi = 3.141592653589793
    <{RealType}> oc[k, l] = real(old_coords[k, l])
    for i
        Y = Y + oc[i, 1]
    end
    for j
        <{RealType}> nc0 = atan2(oc[j, 1], oc[j, 0]) / (pi* 2)
        nc0 = nc0 + 1 if nc0 < 0 else nc0
        nc0 = 1 if nc0 == 0 and Y < 0 else nc0
        new_coords[j, 0] = nc0 * Lx[0]
        new_coords[j, 1] = old_coords[j, 2] * Ly[0]
    end
    """

    cLx = Constant(La)
    cLy = Constant(Lb)

    par_loop((domain, instructions), dx,
             {"new_coords": (new_coordinates, WRITE),
              "old_coords": (old_coordinates, READ),
              "Lx": (cLx, READ),
              "Ly": (cLy, READ)},
             is_loopy_kernel=True)

    if direction == "y":
        # flip x and y coordinates
        operator = np.asarray([[0, 1],
                               [1, 0]])
        new_coordinates.dat.data[:] = np.dot(new_coordinates.dat.data, operator.T)

    return mesh.Mesh(new_coordinates, name=name)
