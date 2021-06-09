# -*- coding: utf-8 -*-
"""Dissipative: Oscillator(Charger)-Qubit(Battery), Time=4pi/g, Target=|0><0||1><1|.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HoQ-z9cEv9sfUwZCVcO2cnFfyeN3_yD6

# Optimization of Dissipative Qubit Reset
"""



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
\newcommand{int}[0]{\text{int}}
\newcommand{opt}[0]{\text{opt}}
\newcommand{tgt}[0]{\text{tgt}}
\newcommand{init}[0]{\text{init}}
\newcommand{lab}[0]{\text{lab}}
\newcommand{rwa}[0]{\text{rwa}}
\newcommand{bra}[1]{\langle#1\vert}
\newcommand{ket}[1]{\vert#1\rangle}
\newcommand{Bra}[1]{\left\langle#1\right\vert}
\newcommand{Ket}[1]{\left\vert#1\right\rangle}
\newcommand{Braket}[2]{\left\langle #1\vphantom{#2} \mid
#2\vphantom{#1}\right\rangle}
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

This example illustrates an optimization in an *open* quantum system,
where the dynamics is governed by the Liouville-von Neumann equation. Hence,
states are represented by density matrices $\op{\rho}(t)$ and the time-evolution
operator is given by a general dynamical map $\DynMap$.

## Define parameters

The system consists of a qubit with Hamiltonian
$\op{H}_{q}(t) = - \frac{\omega_{q}}{2} \op{\sigma}_{z} - \frac{\epsilon(t)}{2} \op{\sigma}_{z}$,
where $\omega_{q}$ is an energy level splitting that can be dynamically adjusted
by the control $\epsilon(t)$. This qubit couples strongly to another two-level
system (TLS) with Hamiltonian $\op{H}_{t} = - \frac{\omega_{t}}{2} \op{\sigma}_{z}$ with
static energy level splitting $\omega_{t}$. The coupling strength between both
systems is given by $J$ with the interaction Hamiltonian given by $\op{H}_{\int}
= J \op{\sigma}_{x} \otimes \op{\sigma}_{x}$.

The Hamiltonian for the system of qubit and TLS is

$$
  \op{H}(t)
    = \op{H}_{q}(t) \otimes \identity_{t}
      + \identity_{q} \otimes \op{H}_{t} + \op{H}_{\int}.
$$

In addition, the TLS is embedded in a heat bath with inverse temperature
$\beta$. The TLS couples to the bath with rate $\kappa$. In order to simulate
the dissipation arising from this coupling, we consider the two Lindblad
operators

$$
\begin{split}
\op{L}_{1} &= \sqrt{\kappa (N_{th}+1)} \identity_{q} \otimes \ket{0}\bra{1} \\
\op{L}_{2} &= \sqrt{\kappa N_{th}} \identity_{q} \otimes \ket{1}\bra{0}
\end{split}
$$

with $N_{th} = 1/(e^{\beta \omega_{t}} - 1)$.
"""

omega = 1  # qubit level splitting
g = 0.2*omega  # qubit-TLS coupling
gamma = 0.05*omega  # TLS decay rate
beta = 1000000000  # inverse bath temperature
T = 4*math.pi/g  # final time
nt = 1000  # number of time steps

"""## Define the Liouvillian

The dynamics of the qubit-TLS system state $\op{\rho}(t)$ is governed by the
Liouville-von Neumann equation

$$
\begin{split}
  \frac{\partial}{\partial t} \op{\rho}(t)
    &= \Liouville(t) \op{\rho}(t) \\
    &= - i \left[\op{H}(t), \op{\rho}(t)\right]
      + \sum_{k=1,2} \left(
            \op{L}_{k} \op{\rho}(t) \op{L}_{k}^\dagger
            - \frac{1}{2}
            \op{L}_{k}^\dagger
            \op{L}_{k} \op{\rho}(t)
            - \frac{1}{2} \op{\rho}(t)
            \op{L}_{k}^\dagger
            \op{L}_{k}
        \right)\,.
\end{split}
$$
"""

def liouvillian(omega, g, gamma, beta,n):
    """Liouvillian for the coupled system of qubit and TLS"""
    A=np.zeros((n,n))
    B=np.zeros((n,n))
    Mu=np.zeros((n,n))
    for i in range (n):
      for j in range (n):
        if j-i==1:
          A[i,j]=0.5
        elif i-j==1:
          B[i,j]=0.5
    Mu=A+B

    H0_q = qutip.Qobj(omega*np.array(np.dot(qutip.create(n),qutip.destroy(n))))
    # drive qubit Hamiltonian
    H1_q = omega*Mu 

    # drift TLS Hamiltonian
    H0_T = omega*0.5*(-qutip.operators.sigmaz()+qutip.qeye(2))

    # Lift Hamiltonians to joint system operators
    H0 = np.kron(H0_q, np.identity(2)) + np.kron(np.identity(n), H0_T)
    H1 = np.kron(H1_q, np.identity(2))

    # qubit-TLS interaction
    H_int =  g*(np.kron(np.array(qutip.create(n)),np.array([[0, 1], [0, 0]]))+np.kron(np.array(qutip.destroy(n)),np.array([[0, 0], [1, 0]])))

    # convert Hamiltonians to QuTiP objects
    H0 = qutip.Qobj(H0 + H_int)
    H1 = qutip.Qobj(H1)

    # Define Lindblad operators
    N = 1.0 / (np.exp(beta * omega) - 1.0)
    
    L=[]
    k=0
    for i in range (0,n-1):
    # Cooling on TLS
      L.append(np.sqrt(gamma * (N + 1)) * np.kron(np.array(qutip.basis(n,i)*qutip.basis(n,i+1).dag()),np.identity(2)))
      L.append(np.sqrt(gamma * N) * np.kron(np.array(qutip.basis(n,i+1)*qutip.basis(n,i).dag()),np.identity(2)))
      L[k]=qutip.Qobj(L[k])
      L[k+1]=qutip.Qobj(L[k+1])
      k=2*(i+1)
    # convert Lindblad operators to QuTiP objects

    # generate the Liouvillian

    L0 = qutip.liouvillian(H=H0, c_ops=L)
    L1 = qutip.liouvillian(H=H1)

    # Shift the qubit and TLS into resonance by default
    eps0 = lambda t, args: 0.000000000001
    return [L0, [L1, eps0]]

"""## Define the optimization target

The initial state of qubit and TLS are assumed to be in thermal equilibrium with
the heat bath (although only the TLS is directly interacting with the bath).
Both states are given by

$$
  \op{\rho}_{\alpha}^{th} =
\frac{e^{x_{\alpha}} \ket{0}\bra{0} + e^{-x_{\alpha}} \ket{1}\bra{1}}{2
\cosh(x_{\alpha})},
  \qquad
  x_{\alpha} = \frac{\omega_{\alpha} \beta}{2},
$$

with $\alpha = q,t$. The initial state of the bipartite system
of qubit and TLS is given by the thermal state
$\op{\rho}_{th} = \op{\rho}_{q}^{th} \otimes \op{\rho}_{t}^{th}$.

Since we are ultimately only interested in the state of the qubit, we define
`trace_A`. It returns the reduced state of the qubit
$\op{\rho}_{q} = \tr_{t}\{\op{\rho}\}$ when passed
the state $\op{\rho}$ of the bipartite system.
"""

def trace_A(rho):
    """Partial trace over the A degrees of freedom"""
    rho_q = np.zeros(shape=(2, 2), dtype=np.complex_)
    rho_q[0, 0] = rho[0, 0] + rho[2, 2] + rho[4, 4]
    rho_q[0, 1] = rho[0, 1] + rho[2, 3]+ rho[4, 5]
    rho_q[1, 0] = rho[1, 0] + rho[3, 2]+ rho[5, 4]
    rho_q[1, 1] = rho[1, 1] + rho[3, 3]+ rho[5,5]
    return qutip.Qobj(rho_q)

"""The target state is (temporarily) the ground state of the bipartite system,
i.e., $\op{\rho}_{\tgt} = \ket{00}\bra{00}$. Note that in the end we will only
optimize the reduced state of the qubit.
"""

#rho_q_trg = np.diag([1, 0,0])
rho_T_trg = np.diag([0, 1])
#rho_trg = np.kron(rho_q_trg, rho_T_trg)
#rho_trg = qutip.Qobj(rho_trg)

"""Next, the list of `objectives` is defined, which contains the initial and target
state and the Liouvillian $\Liouville(t)$ that determines the system dynamics.

In the following, we define the shape function $S(t)$, which we use in order to
ensure a smooth switch on and off in the beginning and end. Note that at times
$t$ where $S(t)$ vanishes, the updates of the field is suppressed.
"""

def S(t):
    """Shape function for the field update"""
    return krotov.shapes.flattop(
        t, t_start=0, t_stop=T, t_rise=0.005 * T, t_fall=0.005 * T, func='sinsq'
    )

"""We re-use this function to also shape the guess control $\epsilon_{0}(t)$ to be
zero at $t=0$ and $t=T$. This is on top of the originally defined constant
value shifting the qubit and TLS into resonance.
"""

def shape_field(eps0):
    """Applies the shape function S(t) to the guess field"""
    eps0_shaped = lambda t, args: eps0(t, args) * S(t)
    return eps0_shaped

"""At last, before heading to the actual optimization below, we assign the shape
function $S(t)$ to the OCT parameters of the control and choose `lambda_a`, a
numerical parameter that controls the field update magnitude in each iteration.

## Simulate the dynamics of the guess field
"""

tlist = np.linspace(0, T, nt)

def plot_pulse(pulse, tlist):
    fig, ax = plt.subplots()
    if callable(pulse):
        pulse = np.array([pulse(t, args=None) for t in tlist])
    ax.plot(tlist, pulse)
    ax.set_xlabel('time')
    ax.set_ylabel('pulse amplitude')
    plt.show(fig)

"""The following plot shows the guess field $\epsilon_{0}(t)$ as a constant that
puts qubit and TLS into resonance, but with a smooth switch-on and switch-off.

We solve the equation of motion for this guess field, storing the expectation
values for the population in the bipartite levels:

The population dynamics of qubit and TLS ground state show that
both are oscillating and especially the qubit's ground state population reaches
a maximal value at intermediate times $t < T$. This maximum is indeed the
maximum that is physically possible. It corresponds to a perfect swap of
the initial qubit and TLS purities. However, we want to reach this maximum at
final time $T$ (not before), so the guess control is not yet working as desired.

## Optimize

Our optimization target is the ground state $\ket{\Psi_{q}^{\tgt}}
= \ket{0}$ of the qubit, irrespective of the state of the TLS. Thus, our
optimization functional reads

$$
  J_T = 1 -
\Braket{\Psi_{q}^{\tgt}}{\tr_{t}\{\op{\rho}(T)\} \,|\; \Psi_{q}^{\tgt}}\,,
$$

and we first define `print_qubit_error`, which prints out the
above functional after each iteration.
"""

def print_qubit_error(**args):
    """Utility function writing the qubit error to screen"""
    taus = []
    for state_T in args['fw_states_T']:
        taus.append(np.real(np.trace(np.dot(state_T,rho_trg))))
    J_T = 1 - np.average(taus)
    print("    qubit error: %.1e" % J_T)
    return J_T

"""In order to minimize the above functional, we need to provide the correct
`chi_constructor` for the Krotov optimization. This is the only place where the
functional (implicitly) enters the optimization.
Given our bipartite system and choice of $J_T$, the equation for
$\op{\chi}(T)$ reads

$$
  \op{\chi}(T)
  =
  \frac{1}{2} \ket{\Psi_{q}^{\tgt}} \bra{\Psi_{q}^{\tgt}} \otimes \op{1}_{2}
  =
  \frac{1}{2} \ket{00}\bra{00} + \frac{1}{2} \ket{01}\bra{01}.
$$
"""

def chis_qubit(fw_states_T, objectives, tau_vals):
    """Calculate chis for the chosen functional"""
    chis = []
    for state_i_T in fw_states_T:
        chi_i = qutip.Qobj(rho_trg)
        chis.append(chi_i)
    return chis

"""We now carry out the optimization for five iterations."""

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
    fig.savefig('FinalEFieldForn=' + str(n) + "tar=01.png",format="PNG")

from numpy import linalg as npla

def eigenvalues(A):
    eigenValues, eigenVectors = npla.eig(A)
    idx = np.argsort(eigenValues)
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return (eigenValues)

# NBVAL_IGNORE_OUTPUT
# the DensityMatrixODEPropagator is not sufficiently exact to guarantee that
# you won't get slightly different results in the optimization when
# running this on different systems
for n in range (95, 105,5):
  L=liouvillian(omega, g, gamma, beta, n)
  rho_q_trg = np.array(qutip.basis(n,0)*qutip.basis(n,0).dag())
  rho_T_trg = np.diag([0, 1])
  rho_trg = np.kron(rho_q_trg, rho_T_trg)
  rho_trg = qutip.Qobj(rho_trg)
  rho_th = qutip.Qobj(np.kron(np.array(qutip.basis(n,0)*qutip.basis(n,0).dag()), np.diag([1,0])))
  objectives = [krotov.Objective(initial_state=rho_th, target=rho_trg, H=L)]
  L[1][1] = shape_field(L[1][1])
  pulse_options = {L[1][1]: dict(lambda_a=1, update_shape=S)}
  opt_result = krotov.optimize_pulses(
      objectives,
      pulse_options,
      tlist,
      iter_stop=20000,
      propagator=krotov.propagators.DensityMatrixODEPropagator(
          atol=1e-10, rtol=1e-8
      ),
      chi_constructor=chis_qubit,
      info_hook=krotov.info_hooks.chain(
          krotov.info_hooks.print_debug_information, print_qubit_error
      ),
      check_convergence=krotov.convergence.Or(
          krotov.convergence.value_below('5e-3', name='J_T'),
          krotov.convergence.check_monotonic_error,
      ),
        store_all_pulses=True,
  )
  print("The following plots are for the number of levels n= ", n)
  plot_iterations(opt_result)
  

  optimized_dynamics = opt_result.optimized_objectives[0].mesolve(
        tlist, e_ops=[]
    )
  Ergotropy=np.zeros(nt)
  Energy=np.zeros(nt)
  time=np.zeros(nt)

  for i in range(0,nt):
    d=np.shape(optimized_dynamics.states[i])[0]//2
    FinalRho=np.trace(np.array(optimized_dynamics.states[i]).reshape(d,2,d,2), axis1=0, axis2=2)
    Rho_f=eigenvalues(FinalRho)[1]*np.array([[1, 0], [0, 0]])+eigenvalues(FinalRho)[0]*np.array([[0, 0], [0, 1]])
    Energy[i]=np.real(np.matrix.trace(omega*np.dot(np.array([[0, 0], [0, 1]]),FinalRho)))
    Ergotropy[i]=-np.real(np.matrix.trace(omega*np.dot(np.array([[0, 0], [0, 1]]),(Rho_f-FinalRho))))
    time[i]=(T/nt)*i

  print("Final energy of the battery is ", Energy[nt-1])
  print("Final ergotropy is ", Ergotropy[nt-1])
  fig2, bx =plt.subplots()
  bx.plot(time,Energy,label='Energy')
  bx.plot(time,Ergotropy,label='Ergotropy')
  bx.set_xlabel("time")
  bx.set_ylabel("Energy, Ergotropy")
  bx.legend()  
  fig2.savefig('Finaln=' + str(n) + "01.png", format="PNG")

"""## Simulate the dynamics of the optimized field

The plot of the optimized field shows that the optimization slightly shifts
the field such that qubit and TLS are no longer perfectly in resonance.

This slight shift of qubit and TLS out of resonance delays the population
oscillations between qubit and TLS ground state such that the qubit ground
state is maximally populated at final time $T$.
"""
