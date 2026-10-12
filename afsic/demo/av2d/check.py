"""Mesh/initial-equilibrium checks against the supplied AV2D source data."""
import sys
from pathlib import Path
import numpy as np
from main import Solid, pressures, read_exodus
source=Path(sys.argv[1])
for filename,ncells,nroots in [('2d_leaflets_sideset.e',590,10),('2d_aorta_sideset.e',3112,0)]:
    s=Solid(source/filename,nroots>0)
    assert len(s.cells)==ncells
    assert s.root_count==nroots
    assert np.max(np.abs(s.jacobians()-1))<1e-10
    s.assemble()
    assert np.max(np.abs(s.force.x.array))<.01, np.max(np.abs(s.force.x.array))
    points,cells,sides=read_exodus(source/filename)
    assert set(sides)=={1,2,3,4}
    assert all(len(np.unique(edge))==2 for edge in sides[4])
    print(filename,'cells',ncells,'root facets',s.root_count,'initial max load',np.max(np.abs(s.force.x.array)))
p=pressures(source/'bc_data.txt')
assert p[-1,0]>=.8512
np.testing.assert_allclose(p[1,1:]/1333.22387415,[.00408678,.01528198],rtol=1e-10)
print('Mesh, side-set, stress-free reference and pressure conversion checks passed')
