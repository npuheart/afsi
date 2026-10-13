"""Mesh/initial-equilibrium checks against the supplied AV2D source data."""
import argparse
from pathlib import Path
import numpy as np
from main import Solid, pressures, read_exodus
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('source',type=Path)
parser.add_argument('--refinement',type=int,choices=[0,1,2],default=0)
a=parser.parse_args()
source=a.source
for filename,ncells,nroots in [('2d_leaflets_sideset.e',590,10),('2d_aorta_sideset.e',3112,0)]:
    s=Solid(source/filename,nroots>0,a.refinement)
    assert len(s.cells)==ncells*4**a.refinement
    assert s.root_count==nroots*2**a.refinement
    assert np.max(np.abs(s.jacobians()-1))<1e-10
    s.assemble()
    assert np.max(np.abs(s.force.x.array))<.01, np.max(np.abs(s.force.x.array))
    points,cells,sides=read_exodus(source/filename)
    assert set(sides)=={1,2,3,4}
    def areas(x,c):
        t=x[c]; e=t[:,1]-t[:,0]; f=t[:,2]-t[:,0]
        return .5*np.abs(e[:,0]*f[:,1]-e[:,1]*f[:,0])
    refined_area=areas(s.ref.reshape(-1,2),s.cells)
    assert refined_area.min()>0
    np.testing.assert_allclose(refined_area.sum(),areas(points,cells).sum(),rtol=1e-12)
    if a.refinement:
        from main import refine_triangles
        _,_,refined_sides=refine_triangles(points,cells,sides,a.refinement)
        for tag in sides:assert len(refined_sides[tag])==len(sides[tag])*2**a.refinement
    assert all(len(np.unique(edge))==2 for edge in sides[4])
    print(filename,'cells',len(s.cells),'root facets',s.root_count,'initial max load',np.max(np.abs(s.force.x.array)))
p=pressures(source/'bc_data.txt')
assert p[-1,0]>=.8512
np.testing.assert_allclose(p[1,1:]/1333.22387415,[.00408678,.01528198],rtol=1e-10)
print('Mesh, side-set, stress-free reference and pressure conversion checks passed')
