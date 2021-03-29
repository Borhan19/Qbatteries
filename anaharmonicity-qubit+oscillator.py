pip install krotov

import qutip as qt
import numpy as np
import scipy
import matplotlib
import matplotlib.pylab as plt
import krotov
import math
from numpy import linalg as npla

def eigenvalues(A):
    eigenValues, eigenVectors = npla.eig(A)
    idx = np.argsort(eigenValues)
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return eigenValues
    
def eigenvectors(A):
    eigenValues, eigenVectors = npla.eig(A)
    idx = np.argsort(eigenValues)
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return eigenVectors

def trace_A(rho):
    """Partial trace over the TLS degrees of freedom"""
    rho_q = np.zeros(shape=(2, 2), dtype=np.complex_)
    rho_q[0, 0] = rho[0, 0] + rho[2, 2]
    rho_q[0, 1] = rho[0, 1] + rho[2, 3]
    rho_q[1, 0] = rho[1, 0] + rho[3, 2]
    rho_q[1, 1] = rho[1, 1] + rho[3, 3]
    return qt.Qobj(rho_q)

def H1_coeff(t, args):
      return  np.exp(-(t*1j*omega))

def H2_coeff(t, args):
      return  np.exp(t*1j*omega)


#For anaharmonicities

N=50
omega=1
Xbb=1
g=0.2*omega
F=0.1*omega
HA=omega/2*(-qt.sigmaz()+qt.qeye(2))
HB=omega*qt.create(N)*qt.destroy(N)- Xbb/2*(qt.create(N)*qt.destroy(N)*qt.create(N)*qt.destroy(N))
HAB=g*(qt.tensor(qt.create(N),qt.destroy(N))+qt.tensor(qt.destroy(N), qt.create(N)))
H1=qt.tensor(F*qt.create(2),qt.qeye(2))
H2=qt.tensor(F*qt.destroy(2),qt.qeye(2))
H0=qt.tensor(HA, qt.qeye(N))+qt.tensor(qt.qeye(2),HB)+HAB
H=[H0, [H1, H1_coeff], [H2, H2_coeff]]
t=np.linspace(0,30*math.pi/g,10000)
psi0=qt.basis(N**2,0)
output = qt.mesolve(H, psi0, t, [qt.tensor(np.sqrt(0.05*omega)*qt.destroy(2),qt.qeye(N))], [])
Energy=np.zeros(10000)
Ergotropy=np.zeros(10000)
v=eigenvectors(HB)
EntropyB=np.zeros(10000)

for i in range(0,10000):
  A=np.array(output.states[i])
  FinalRho=np.trace(A.reshape(2,N,2,N), axis1=0, axis2=2)
  Rho_f=np.zeros((N,N))
  for j in range(0,N):
      Rho_f=eigenvalues(FinalRho)[N-1-j]*v[:,j]+Rho_f
  Energy[i-1]=np.real(np.matrix.trace(omega*np.dot(np.array(HB),FinalRho)))
  Ergotropy[i-1]=-np.real(np.matrix.trace(omega*np.dot(np.array(HB),(Rho_f-FinalRho)))) 
  EntropyB[i-1]=qt.entropy_vn(qt.Qobj(FinalRho),2)
  
  
plt.figure()
plt.plot(t,Energy/omega,label="Energy/omega")
plt.plot(t,Ergotropy/omega,label="Ergotropy/omega")
plt.xlabel("Time")
plt.title("Anaharmonicities")
plt.legend()

plt.figure()
plt.plot(t,EntropyB)
plt.xlabel("Time")
plt.title("Entropy")
plt.legend()
plt.show()

for alpha in np.arange(0,100,5):  
  Modulus=np.zeros(10000)
  for i in range (0,10000):
    A=np.array(output.states[i])
    FinalRho=np.trace(A.reshape(2,N,2,N), axis1=0, axis2=2)
    Cat=np.array(qt.coherent(N,alpha)-qt.coherent(N,-alpha))
    Cat=Cat/(np.sqrt(np.dot(np.transpose(Cat),Cat)))
    if alpha==0:
      Cat=np.array(qt.coherent(N,alpha))
      Cat=Cat/(np.sqrt(np.dot(np.transpose(Cat),Cat)))
    Modulus[i]=np.real(np.trace(np.dot(FinalRho,np.dot(Cat,np.transpose(Cat)))))
  print("The plot is for alpha =",alpha)
  plt.figure()
  plt.plot(t,Modulus)
  plt.xlabel("Time")
  plt.title("Fidelity")
  plt.show()
  plt.savefig("Fidelity for alpha", str(alpha) + ".png", format="PNG")

