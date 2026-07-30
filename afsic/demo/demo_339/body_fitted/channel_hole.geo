SetFactory("OpenCASCADE");

// Channel with a cylindrical hole — body-fitted mesh
// Fluid domain = rectangle minus cylinder disk.
// Physical Curve tags: 11=inlet, 12=outlet, 13=bottom, 14=top, 15=cylinder
// Physical Surface: "fluid"

Lx = 2.2;
Ly = 0.41;
cx = 0.2;
cy = 0.2;
R  = 0.05;
eps = 1e-6;

Rectangle(1) = {0, 0, 0, Lx, Ly};
Disk(2) = {cx, cy, 0, R};

// Keep a copy of the disk
Duplicata { Surface{2}; }
BooleanDifference(4) = { Surface{1}; Delete; }{ Surface{3}; Delete; };

Physical Surface("fluid") = {4};

// Boundary curves
inlet_c[]   = Curve In BoundingBox{ -eps, -eps, -eps,  eps,   Ly+eps, eps };
outlet_c[]  = Curve In BoundingBox{ Lx-eps, -eps, -eps, Lx+eps, Ly+eps, eps };
top_c[]     = Curve In BoundingBox{ -eps, Ly-eps, -eps, Lx+eps, Ly+eps, eps };
bottom_c[]  = Curve In BoundingBox{ -eps, -eps, -eps, Lx+eps, eps,    eps };
cylinder_c[]= Curve In BoundingBox{ cx-R-eps, cy-R-eps, -eps, cx+R+eps, cy+R+eps, eps };

Physical Curve("inlet")    = inlet_c[];
Physical Curve("outlet")   = outlet_c[];
Physical Curve("top")      = top_c[];
Physical Curve("bottom")   = bottom_c[];
Physical Curve("cylinder") = cylinder_c[];

Mesh.MeshSizeMax = 0.02;
Mesh.MeshSizeMin = 0.002;  // finer near cylinder
