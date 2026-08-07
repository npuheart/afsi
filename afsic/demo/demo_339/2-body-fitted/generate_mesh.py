"""Generate body-fitted mesh: channel with cylindrical hole → .msh file."""

import gmsh

gmsh.initialize()
gmsh.model.add("channel_hole")
gmsh.merge("channel_hole.geo")
gmsh.model.mesh.generate(dim=2)
gmsh.write("channel_hole.msh")
gmsh.finalize()
print("channel_hole.msh written")
