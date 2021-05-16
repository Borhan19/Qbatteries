# -*- coding: utf-8 -*-
"""Non-dissipative, Time=pi/g, g=0.2*omega, Target=|0><0| |1><1|, 4cells.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cMJtv02Uk5UNDvlGXcKBzysO2Vq4tNV3

# Optimization of a State-to-State Transfer for a Quantum Charger-Battery Driven by Laser Field Using Krotov's Method
"""

pip install krotov

# NBVAL_IGNORE_OUTPUT
import qutip
import numpy as np
import scipy
import matplotlib
import matplotlib.pylab as plt
import krotov
import math

"""$\newcommand{tr}[0]{\operatorname{tr}}
\newcommand{diag}[0]{\operatorname{diag}}
\newcommand{abs}[0]{\operatorname{abs}}
\newcommand{pop}[0]{\operatorname{pop}}
\newcommand{aux}[0]{\text{aux}}
\newcommand{opt}[0]{\text{opt}}
\newcommand{tgt}[0]{\text{tgt}}
\newcommand{init}[0]{\text{init}}
\newcommand{lab}[0]{\text{lab}}
\newcommand{rwa}[0]{\text{rwa}}
\newcommand{bra}[1]{\langle#1\vert}
\newcommand{ket}[1]{\vert#1\rangle}
\newcommand{Bra}[1]{\left\langle#1\right\vert}
\newcommand{Ket}[1]{\left\vert#1\right\rangle}
\newcommand{Braket}[2]{\left\langle #1\vphantom{#2}\mid{#2}\vphantom{#1}\right\rangle}
\newcommand{op}[1]{\hat{#1}}
\newcommand{Op}[1]{\hat{#1}}
\newcommand{dd}[0]{\,\text{d}}
\newcommand{Liouville}[0]{\mathcal{L}}
\newcommand{DynMap}[0]{\mathcal{E}}
\newcommand{identity}[0]{\mathbf{1}}
\newcommand{Norm}[1]{\lVert#1\rVert}
\newcommand{Abs}[1]{\left\vert#1\right\vert}
\newcommand{avg}[1]{\langle#1\rangle}
\newcommand{Avg}[1]{\left\langle#1\right\rangle}
\newcommand{AbsSq}[1]{\left\vert#1\right\vert^2}
\newcommand{Re}[0]{\operatorname{Re}}
\newcommand{Im}[0]{\operatorname{Im}}$

This first example illustrates the basic use of the `krotov` package by solving
a simple canonical optimization problem: the transfer of population in a two
level system.

## Hamiltonian
"""

proj0 = qutip.ket2dm(qutip.tensor(qutip.ket("0"),qutip.ket("0")))
proj1 = qutip.ket2dm(qutip.tensor(qutip.ket("0"),qutip.ket("1")))
proj2 = qutip.ket2dm(qutip.tensor(qutip.ket("1"),qutip.ket("0")))
proj3 = qutip.ket2dm(qutip.tensor(qutip.ket("1"),qutip.ket("1")))
omega=1
g=0.2*omega
ampl0=1
T=math.pi/g
nt=1000
tlist = np.linspace(0,T, nt)
def hamiltonian(omega, ampl0, g):
    
    
    
    H0_q = omega*0.5*(-qutip.operators.sigmaz()+qutip.qeye(2))
    # drive qubit Hamiltonian
    H1_q = -0.5*qutip.operators.sigmax()

    # drift TLS Hamiltonian
    H0_T = qutip.tensor(omega*0.5*(-qutip.operators.sigmaz()+qutip.qeye(2)), qutip.qeye(2), qutip.qeye(2), qutip.qeye(2))\
                        +qutip.tensor(qutip.qeye(2), omega*0.5*(-qutip.operators.sigmaz()+qutip.qeye(2)), qutip.qeye(2), qutip.qeye(2))\
                        +qutip.tensor(qutip.qeye(2), qutip.qeye(2), omega*0.5*(-qutip.operators.sigmaz()+qutip.qeye(2)), qutip.qeye(2))\
                        +qutip.tensor(qutip.qeye(2), qutip.qeye(2), qutip.qeye(2), omega*0.5*(-qutip.operators.sigmaz()+qutip.qeye(2)))

    # Lift Hamiltonians to joint system operators
    H0 = qutip.tensor(H0_q, qutip.qeye(2), qutip.qeye(2), qutip.qeye(2),qutip.qeye(2)) + qutip.tensor(qutip.qeye(2), H0_T)
    H1 = qutip.tensor(H1_q, qutip.qeye(2), qutip.qeye(2), qutip.qeye(2),qutip.qeye(2))

    # qubit-TLS interaction
    H_int =  g*(qutip.tensor(qutip.destroy(2),qutip.create(2),qutip.qeye(2),qutip.qeye(2),qutip.qeye(2))\
                +qutip.tensor(qutip.create(2),qutip.destroy(2),qutip.qeye(2),qutip.qeye(2),qutip.qeye(2))\
                +qutip.tensor(qutip.destroy(2),qutip.qeye(2),qutip.create(2),qutip.qeye(2),qutip.qeye(2))\
                +qutip.tensor(qutip.create(2),qutip.qeye(2),qutip.destroy(2),qutip.qeye(2),qutip.qeye(2))\
                +qutip.tensor(qutip.destroy(2),qutip.qeye(2),qutip.qeye(2),qutip.create(2),qutip.qeye(2))\
                +qutip.tensor(qutip.create(2),qutip.qeye(2),qutip.qeye(2),qutip.destroy(2),qutip.qeye(2))\
                +qutip.tensor(qutip.destroy(2),qutip.qeye(2),qutip.qeye(2),qutip.qeye(2),qutip.create(2))\
                +qutip.tensor(qutip.create(2),qutip.qeye(2),qutip.qeye(2),qutip.qeye(2),qutip.destroy(2)))

    # convert Hamiltonians to QuTiP objects
    H0 = qutip.Qobj(H0 + H_int)
    H1 = qutip.Qobj(H1)

    def guess_control(t, args):
        return ampl0 * krotov.shapes.flattop(
            t, t_start=0, t_stop=T, t_rise=0.5, func="blackman"
        )

    return [H0, [H1, guess_control]]
def S(t):
    """Shape function for the field update"""
    return krotov.shapes.flattop(
        t, t_start=0, t_stop=T, t_rise=0.05 * T, t_fall=0.05 * T, func='sinsq'
    )
def plot_iterations(opt_result):
    """Plot the control fields in population dynamics over all iterations.

    This depends on ``store_all_pulses=True`` in the call to
    `optimize_pulses`.
    """

    fig, [ax_ctr,ax] = plt.subplots(nrows=2, figsize=(4, 5))
    n_iters = len(opt_result.iters)
    EEnergy=np.zeros(nt)
    for (iteration, pulses) in zip(opt_result.iters, opt_result.all_pulses):
        controls = [
            krotov.conversions.pulse_onto_tlist(pulse)
            for pulse in pulses
        ]
        objectives = opt_result.objectives_with_controls(controls)
        dynamics = objectives[0].mesolve(
            opt_result.tlist, e_ops=[]
        )
        if iteration == 0:
            ls = '--'  # dashed
            alpha = 1  # full opacity
            ctr_label = 'guess'
            pop_labels = ['0 (guess)', '1 (guess)']
        elif iteration == opt_result.iters[-1]:
            ls = '-'  # solid
            alpha = 1  # full opacity
            ctr_label = 'optimized'
            pop_labels = ['0 (optimized)', '1 (optimized)']
        else:
            ls = '-'  # solid
            alpha = 0.5 * float(iteration) / float(n_iters)  # max 50%
            ctr_label = None
            pop_labels = [None, None]
        ax_ctr.plot(
            dynamics.times,
            controls[0],
            label=ctr_label,
            color='black',
            ls=ls,
            alpha=alpha,
        )
    EField=np.transpose(np.array(opt_result.optimized_controls))
    EEnergy[0]=(np.square(EField[0]))*(T/nt)
    a=0
    for i in range (1,nt):
      a+=np.square(EField[i-1])
      EEnergy[i]=(np.square(EField[i])+a)*(T/nt)
      
    
    ax.plot(tlist,np.transpose(EEnergy))
    plt.legend()
    plt.show(fig)
    
    
    



H = hamiltonian(omega,ampl0,g)
pulse_options = {
    H[1][1]: dict(lambda_a=0.5, update_shape=S)
  }
objectives = [
    krotov.Objective(
        initial_state=qutip.tensor(qutip.ket("0"),qutip.ket("0"),qutip.ket("0"),qutip.ket("0"),qutip.ket("0")), target=qutip.tensor(qutip.ket("0"),qutip.ket("1"),qutip.ket("1"),qutip.ket("1"),qutip.ket("1")), H=H
      )
  ]

  

opt_result = krotov.optimize_pulses(
  objectives,
  pulse_options=pulse_options,
  tlist=tlist,
  propagator=krotov.propagators.expm,
  chi_constructor=krotov.functionals.chis_ss,
  info_hook=krotov.info_hooks.print_table(J_T=krotov.functionals.J_T_ss),
  check_convergence=krotov.convergence.Or(
      krotov.convergence.value_below('5e-3', name='J_T'),
      krotov.convergence.check_monotonic_error,
  ),
    store_all_pulses=True,
)
plot_iterations(opt_result)

from numpy import linalg as npla

def eigenvalues(A):
    eigenValues, eigenVectors = npla.eig(A)
    idx = np.argsort(eigenValues)
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return (eigenValues)

optimized_dynamics = opt_result.optimized_objectives[0].mesolve(
      tlist, e_ops=[]
  )
Ergotropy=np.zeros(nt)
Energy=np.zeros(nt)
time=np.zeros(nt)
Power=np.zeros(nt)
a=0
for i in range(0,nt):
  FinalStateB = np.trace(np.array(optimized_dynamics.states[i]*optimized_dynamics.states[i].dag()).reshape(2,2,2,2,2,2,2,2,2,2), axis1=0, axis2=5)
  FinalStateCell1=np.trace(np.array(FinalStateB).reshape(2,8,2,8), axis1=1, axis2=3)
  PreFinalStateCell2=np.trace(np.array(FinalStateB).reshape(2,2,4,2,2,4), axis1=2, axis2=5)
  FinalStateCell2=np.trace(np.array(PreFinalStateCell2).reshape(2,2,2,2), axis1=0, axis2=2)
  PreFinalStateCell3=np.trace(np.array(FinalStateB).reshape(4,2,2,4,2,2), axis1=0, axis2=3)
  FinalStateCell3=np.trace(np.array(PreFinalStateCell2).reshape(2,2,2,2), axis1=1, axis2=3)
  FinalStateCell4=np.trace(np.array(FinalStateB).reshape(8,2,8,2), axis1=0, axis2=2)    

  Rho_fCell1=eigenvalues(FinalStateCell1)[1]*np.array([[1, 0], [0, 0]])+eigenvalues(FinalStateCell1)[0]*np.array([[0, 0], [0, 1]])
  Rho_fCell2=eigenvalues(FinalStateCell2)[1]*np.array([[1, 0], [0, 0]])+eigenvalues(FinalStateCell2)[0]*np.array([[0, 0], [0, 1]])
  Rho_fCell3=eigenvalues(FinalStateCell3)[1]*np.array([[1, 0], [0, 0]])+eigenvalues(FinalStateCell3)[0]*np.array([[0, 0], [0, 1]])
  Rho_fCell4=eigenvalues(FinalStateCell4)[1]*np.array([[1, 0], [0, 0]])+eigenvalues(FinalStateCell4)[0]*np.array([[0, 0], [0, 1]])
  Energy[i]=np.real(np.matrix.trace(omega*np.dot(np.array([[0, 0], [0, 1]]),FinalStateCell4)))+np.real(np.matrix.trace(omega*np.dot(np.array([[0, 0], [0, 1]]),FinalStateCell3)))+np.real(np.matrix.trace(omega*np.dot(np.array([[0, 0], [0, 1]]),FinalStateCell1)))+np.real(np.matrix.trace(omega*np.dot(np.array([[0, 0], [0, 1]]),FinalStateCell2)))
  Ergotropy[i]=-np.real(np.matrix.trace(omega*np.dot(np.array([[0, 0], [0, 1]]),(Rho_fCell4-FinalStateCell4))))-np.real(np.matrix.trace(omega*np.dot(np.array([[0, 0], [0, 1]]),(Rho_fCell3-FinalStateCell3))))-np.real(np.matrix.trace(omega*np.dot(np.array([[0, 0], [0, 1]]),(Rho_fCell1-FinalStateCell1))))-np.real(np.matrix.trace(omega*np.dot(np.array([[0, 0], [0, 1]]),(Rho_fCell2-FinalStateCell2))))
  a+=1/T*(Energy[i]*T/nt) 
  Power[i]=1/T*(Energy[i]*T/nt) + a 
  time[i]=(T/nt)*i

print(np.argmax(Power))
plt.plot(time,Energy,label='Energy')
plt.plot(time,Ergotropy,label='Ergotropy')
plt.plot(time,Power,label='Power')
plt.xlabel("Time")
plt.ylabel("Energy, Ergotropy")
plt.legend()  
plt.show()

def plot_pulse(pulse, tlist):
    fig, ax = plt.subplots()
    if callable(pulse):
        pulse = np.array([pulse(t, args=None) for t in tlist])
    ax.plot(tlist, pulse)
    ax.set_xlabel('Time')
    ax.set_ylabel('Pulse Amplitude')
    plt.show(fig)

plot_pulse(opt_result.optimized_controls[0], tlist)