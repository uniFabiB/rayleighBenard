n = 256;	// ungefähr

lc = 10/n;

Mesh.MeshSizeMin = lc/10;
Mesh.MeshSizeMax = lc;

//+
Point(1) = {-3, -4, 0, lc};
//+
Point(2) = {-3.5, -1, 0, lc};
//+
Point(3) = {5, 1, 0, lc};
//+
Point(4) = {7.5, -4, 0, lc};
//+
Point(5) = {2.5, -6, 0, lc};
//+
Point(6) = {2.5, -3, 0, lc};
//+
Spline(11) = {1:6,1};

Curve Loop(101) = {11};

Plane Surface(1001) = {101};

Physical Surface(2001) = {1001};
