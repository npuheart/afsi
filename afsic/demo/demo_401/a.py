import gmsh, math
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("sperm3d")
occ = gmsh.model.occ

# ---- parameters ----
R, r       = 0.05, 0.01
L1, L2, L3 = 0.01, 0.08, 0.21
ms_h, ms_t = 0.005, 0.008
x0  = math.sqrt(R*R - r*r)
xs  = [x0, x0+L1, x0+L1+L2, x0+L1+L2+L3]   # segment x-boundaries
Ls  = [L1, L2, L3]

# ---- geometry ----
s  = occ.addSphere(0, 0, 0, R)
c1 = occ.addCylinder(xs[0], 0, 0, L1, 0, 0, r)
c2 = occ.addCylinder(xs[1], 0, 0, L2, 0, 0, r)
c3 = occ.addCylinder(xs[2], 0, 0, L3, 0, 0, r)

# Fragments keeps inter-segment interfaces -> conformal mesh, 3 distinct laterals
out, _ = occ.fragment([(3, s)], [(3, c1), (3, c2), (3, c3)])
occ.synchronize()

all_vols = [t for (d, t) in gmsh.model.getEntities(3)]

# ---- classify EXTERIOR surfaces topologically ----
TIP, HEAD = 16, 15
seg_tag = {0: 17, 1: 18, 2: 19}           # neck / mid / long
groups = {15: [], 16: [], 17: [], 18: [], 19: []}
tol = 1e-6

for (d, sft) in gmsh.model.getEntities(2):
    adj_vols = gmsh.model.getAdjacencies(2, sft)[0]
    if len(adj_vols) != 1:                 # interior interface -> skip
        continue
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, sft)
    rad = max(abs(ymin), abs(ymax), abs(zmin), abs(zmax))
    dx  = xmax - xmin
    if rad > r + tol:                      # reaches sphere radius -> head
        groups[HEAD].append(sft)
    elif dx < tol:                         # flat disk perpendicular to x -> tip cap
        groups[TIP].append(sft)
    else:                                  # lateral: locate by its midpoint
        xc = 0.5*(xmin + xmax)
        for i in range(3):
            if xs[i] - tol <= xc <= xs[i+1] + tol:
                groups[seg_tag[i]].append(sft); break

for tag, surfs in groups.items():
    gmsh.model.addPhysicalGroup(2, surfs, tag)
gmsh.model.addPhysicalGroup(3, all_vols, 1)

# ---- mesh sizing ----
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), ms_t)
f = gmsh.model.mesh.field
f.add("Box", 1)
f.setNumber(1, "VIn", ms_h); f.setNumber(1, "VOut", ms_t)
f.setNumber(1, "XMin", -R-1e-3); f.setNumber(1, "XMax", 0.0)
f.setNumber(1, "YMin", -R-1e-3); f.setNumber(1, "YMax", R+1e-3)
f.setNumber(1, "ZMin", -R-1e-3); f.setNumber(1, "ZMax", R+1e-3)
f.setAsBackgroundMesh(1)

# ---- report (then you'd call gmsh.model.mesh.generate(3); gmsh.write("sperm3d.msh")) ----
for tag in (15,16,17,18,19):
    surfs = groups[tag]
    desc=[]
    for sf in surfs:
        xmin,_,_,xmax,_,_=gmsh.model.getBoundingBox(2,sf)
        desc.append(f"{sf}[dx={xmax-xmin:.3f}]")
    print(f"Physical Surface {tag}: {desc}")
print("Physical Volume 1:", all_vols)
gmsh.model.mesh.generate(3)
gmsh.write("sperm3d.msh")
gmsh.finalize()