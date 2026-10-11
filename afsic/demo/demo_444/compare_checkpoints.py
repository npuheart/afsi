"""Compare every saved field in two equivalent Bonet trajectories."""
import sys
import numpy as np
with np.load(sys.argv[1]) as a, np.load(sys.argv[2]) as b:
    assert set(a.files)==set(b.files)
    for key in a.files:
        assert a[key].shape==b[key].shape
        error=float(np.max(np.abs(a[key]-b[key])))
        assert np.array_equal(a[key],b[key]), (key,error)
        print(f'{key}: bitwise equal (max difference {error})')
print('PASS: complete checkpoint states are bitwise identical')
