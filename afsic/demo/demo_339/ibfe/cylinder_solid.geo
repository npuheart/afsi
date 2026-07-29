SetFactory("OpenCASCADE");

// ============================================================
// Solid disk (cylinder) for IB-FE coupling
// Center (0.2, 0.2), radius 0.05  (Turek cylinder dimensions)
// Physical Surface: "disk" (tag 1)
// Physical Curve:   "boundary" (tag 10)
// ============================================================

cx = 0.2;
cy = 0.2;
R  = 0.05;

Disk(1) = {cx, cy, 0, R};

Physical Surface("disk") = {1};

// The boundary curve of the disk
boundary_c[] = Curve In BoundingBox{ cx-R-1e-6, cy-R-1e-6, -1e-6,
                                      cx+R+1e-6, cy+R+1e-6,  1e-6 };
Physical Curve("boundary") = boundary_c[];

// Fine mesh inside the disk for accurate stress computation
Mesh.MeshSizeMax = 0.005;
Mesh.MeshSizeMin = 0.002;
