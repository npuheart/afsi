# import afsic as m

# def test_add():
#     assert m.add(1, 2) == 3

# import os
# from afsic import EmailInfo, send_email

# info = EmailInfo()
# info.smtp_url = "smtp://smtp.qq.com:587"
# info.username = "499908174@qq.com"
# info.password = os.getenv("SMTP_QQ") 
# info.From = "499908174@qq.com"
# info.to = "mapengfei@mail.nwpu.edu.cn"
# info.subject = "Test"
# info.body = "Hello from Python"
# info.is_html = False
# send_email(info)


# # 流体求解器
# from afsic import IPCSSolver



# # 时间步长管理
# from afsic import TimeManager, swanlab_init, swanlab_upload
# from mpi4py import MPI
# import requests
# import time

# project_name = "afsic"
# config = {"key": "value"}
# data_log = {"time" : time.time()}
# experiment_name = requests.get(f"http://counter.pengfeima.cn/{project_name}").text if MPI.COMM_WORLD.rank == 0 else None
# experiment_name = MPI.COMM_WORLD.bcast(experiment_name, root=0)
# swanlab_init(project_name,experiment_name,config)

# swanlab_upload(0.1, data_log, capacity=10, pressure_left=20)

# # 添加流固耦合模块



from afsic import coupling, IBMesh

Nx = 32
Ny = 30
coupling()
ibmesh = IBMesh(0.0,1.0, 0.0,1.0, Nx,Ny,2)


# IBMesh 
from mpi4py import MPI
import ufl
import dolfinx
from dolfinx.mesh import CellType, GhostMode
from basix.ufl import element
import numpy as np
# Create mesh
from dolfinx.fem import (Function, functionspace,
                         dirichletbc, locate_dofs_topological)
mesh = dolfinx.mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(Nx, Ny),
    cell_type=CellType.triangle,
    # cell_type=CellType.quadrilateral,
    ghost_mode=GhostMode.shared_facet,
)

x = ufl.SpatialCoordinate(mesh)
v_cg2 = element("Lagrange", mesh.topology.cell_name(),
                2, shape=(mesh.geometry.dim, ))
V = functionspace(mesh, v_cg2)
coords = Function(V)
coords.interpolate(lambda x: np.array([x[0], x[1]])) 

ibmesh.build_map(coords._cpp_object)
ibmesh.evaluate(0.5,0.5, coords._cpp_object)

