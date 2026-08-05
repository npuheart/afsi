"""demo_422 mesh generation (background cavity + gmsh disk)."""
import os
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

import gmsh
import dolfinx
from dolfinx import mesh
from dolfinx.io import gmsh as gmshio

from config import comm, rank

def create_fluid_mesh(cfg):
    """Background (Eulerian) square cavity on triangles (P2/P1)."""
    msh = mesh.create_rectangle(
        comm, ((0.0, 0.0), (1.0, 1.0)),
        (cfg["Nx"], cfg["Ny"]), cell_type=mesh.CellType.triangle,
    )
    msh.topology.create_connectivity(msh.topology.dim, 0)
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    return msh


def create_solid_mesh(cfg):
    """Solid (Lagrangian) disk via gmsh (reference configuration)."""
    gmsh.initialize()
    gmsh.model.add("disk")
    d = gmsh.model.occ.addDisk(cfg["cx"], cfg["cy"], 0.0, cfg["R"], cfg["R"])
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [d], tag=1)
    gmsh.option.setNumber("Mesh.MeshSizeMax", cfg["solid_h"])
    gmsh.option.setNumber("Mesh.MeshSizeMin", cfg["solid_h"])
    gmsh.model.mesh.generate(2)
    smsh = gmshio.model_to_mesh(gmsh.model, comm, 0, gdim=2).mesh
    gmsh.finalize()
    smsh.topology.create_connectivity(smsh.topology.dim, 0)
    return smsh
