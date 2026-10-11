"""Independent NumPy checks of modified Mooney-Rivlin stress and energy."""
import numpy as np

def energy(F):
    J=np.linalg.det(F); C=F.T@F; I1=np.trace(C)
    I2=(I1*I1-np.trace(C@C))/2
    return 9000*(J**(-2/3)*I1-3)+9000*(J**(-4/3)*I2-3)+84000*np.log(J)**2

def stress(F):
    J=np.linalg.det(F); C=F.T@F; I1=np.trace(C)
    I2=(I1*I1-np.trace(C@C))/2; H=np.linalg.inv(F).T
    return 18000*J**(-2/3)*(F-I1/3*H)+18000*J**(-4/3)*(I1*F-F@C-2*I2/3*H)+168000*np.log(J)*H

assert np.allclose(stress(np.eye(3)),0)
a=.73; R=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
assert abs(energy(R))<1e-8 and np.max(np.abs(stress(R)))<1e-8
F=np.array([[1.1,.13,.04],[.02,.93,.06],[0,.03,1.02]])
assert np.isclose(energy(R@F),energy(F),rtol=1e-12)
assert np.allclose(stress(R@F),R@stress(F),rtol=1e-12,atol=1e-8)
eps=1e-6; numerical=np.empty((3,3))
for i in range(3):
    for j in range(3):
        E=np.zeros((3,3)); E[i,j]=eps
        numerical[i,j]=(energy(F+E)-energy(F-E))/(2*eps)
error=np.max(np.abs(numerical-stress(F)))
assert error<1e-4, error
S=np.eye(3); S[0,1]=.2
assert np.isclose(energy(S),18000*.2**2)
print(f'PASS: identity, rigid rotation, objectivity, simple shear, energy derivative; max FD error={error:.3g}')
