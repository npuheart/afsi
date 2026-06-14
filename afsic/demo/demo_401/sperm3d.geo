// Sperm 3D geometry: sphere head + cylindrical tail
// Units: can be rescaled in Python
//
// Physical Surface tags:
//   15 = nose cap of sphere (fixed/anchored)
//   16 = tail tip disk (driven)
//   17 = tail lateral surface (flagellum)
// Physical Volume:
//    1 = full body

SetFactory("OpenCASCADE");

R  = 0.05;   // sphere radius
r  = 0.01;  // tail (cylinder) radius
L  = 0.30;   // tail length
ms_h = 0.005; // mesh size head
ms_t = 0.008; // mesh size tail

// ── Head: sphere ─────────────────────────────────────────────
Sphere(1) = {0, 0, 0, R};

// ── Tail: cylinder (aligned along +x from sphere surface) ────
// Start at x = sqrt(R²-r²) so the cylinder base sits flush inside the sphere
x0 = Sqrt(R*R - r*r);
Cylinder(2) = {x0, 0, 0,  L, 0, 0,  r};

// ── Boolean: fuse sphere and cylinder ────────────────────────
BooleanUnion(3) = { Volume{1}; Delete; }{ Volume{2}; Delete; };

// ── Mesh sizes ───────────────────────────────────────────────
// Apply via fields or characteristic length on points
MeshSize{ PointsOf{ Volume{3}; } } = ms_t;

// ── Identify surfaces ─────────────────────────────────────────
// After BooleanUnion the surface tags may be renumbered;
// use bounding-box queries to tag them.

// Nose: part of sphere with x < -R/2
Field[1] = Box;
Field[1].VIn  = ms_h;
Field[1].VOut = ms_t;
Field[1].XMin = -R - 0.001;
Field[1].XMax = -R/2;
Field[1].YMin = -R - 0.001;
Field[1].YMax =  R + 0.001;
Field[1].ZMin = -R - 0.001;
Field[1].ZMax =  R + 0.001;
Background Field = 1;

// Tag surfaces by position
nose_surfaces[] = Surface In BoundingBox{-R-0.001, -R-0.001, -R-0.001,
                                          -R/2,      R+0.001,  R+0.001};
tail_tip[]      = Surface In BoundingBox{ x0+L-0.001, -r-0.001, -r-0.001,
                                           x0+L+0.001,  r+0.001,  r+0.001};
tail_lateral[]  = Surface In BoundingBox{ x0-0.001, -r-0.001, -r-0.001,
                                           x0+L+0.001, r+0.001,  r+0.001};

Physical Surface(15) = {nose_surfaces[]};
Physical Surface(16) = {tail_tip[]};
Physical Surface(17) = {tail_lateral[]};
Physical Volume(1)   = {3};
