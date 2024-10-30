from brian2 import *
import numpy as np
import random

from typing import Optional
from numpy.typing import NDArray


def uniform_or_probability(p: Optional[float] = None, w: float = 1, binary=False):
    if p is None:
        return random.uniform(0, 1) * w
    elif random.uniform(0, 1) < p:
        if binary:
            return w
        else:
            return random.uniform(0, 1) * w
    else:
        return 0


def generate_connectivity_matrix(
        nExcitation: int, nInhibition: int,
        pEE: Optional[float] = None, wEE: Optional[float] = None,
        pEI: Optional[float] = None, wEI: Optional[float] = None,
        pIE: Optional[float] = None, wIE: Optional[float] = None,
        pII: Optional[float] = None, wII: Optional[float] = None) -> NDArray[float64]:
    """Generates an excitation-inhibition connectivity matrix

    The matrix will have the following structure where E represents excitation
    and I represents inhibition:

    | EE | EI |
    | IE | II |

    Arguments:
        nExcitation -- Number of excitatory neurons.
        nInhibition -- Number of inhibitory neurons.

    Keyword Arguments:
        pEE -- Probability of a connection between excitatory neurons. (default: {None})
        wEE -- Weight of connection between excitatory neurons. (default: {None})
        pEI -- Probability of a connection between excitatory and inhibitory neurons. (default: {None})
        wEI -- Weight of connection between excitatory and inhibitory neurons. (default: {None})
        pIE -- Probability of a connection between inhibitory and excitatory neurons. (default: {None})
        wIE -- Weight of connection between inhibitory and excitatory neurons. (default: {None})
        pII -- Probability of a connection between inhibitory neurons. (default: {None})
        wII -- Weight of connection between inhibitory neurons. (default: {None})

    Returns:
        The connectivity matrix.
    """
    n = nExcitation + nInhibition
    mat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i is j:
                continue

            if i < nExcitation and j < nExcitation:
                mat[i, j] = uniform_or_probability(p=pEE, w=wEE)
            elif i >= nExcitation and j < nExcitation:
                mat[i, j] = uniform_or_probability(p=pEI, w=wEI)
            elif i < nExcitation and j >= nExcitation:
                mat[i, j] = uniform_or_probability(p=pIE, w=wIE)
            else:
                mat[i, j] = uniform_or_probability(p=pII, w=wII)

    return mat


class Model(object):
    def __init__(self, N=100, pExcitation=0.6, **kwargs):
        self.N = N

        self.nExcitation = round(self.N * pExcitation)
        self.nInhibition = self.N - self.nExcitation

        self.mat = generate_connectivity_matrix(
            self.nExcitation, self.nInhibition, **kwargs)

        params = {
            "Vt": -20*mV,  # spiking threshold
            "dt": 0.025 * ms,  # simulation timestep

            # Reversal Potentials
            "E_leak": -54.4 * mV,
            "E_Na":  55. * mV,
            "E_K": -77. * mV,

            # Conductances
            "g_leak":   300. * uS / cm ** 2,
            "gbar_Na": 120. * mS / cm ** 2,
            "gbar_K":  36. * mS / cm ** 2,

            # Membrane Capacitance
            "Cm": 1. * uF / cm ** 2,
        }

        # Equations:
        eqs_V = """
        dv/dt = (I - I_leak - I_Na - I_K )/Cm : volt
        """

        # note: below we broke down the equations in conductance and current equations
        # we need this to have access to the conductances via Brian's state monitor function

        eqs_cond = """
        g_Na = gbar_Na*(m**3)*h           : siemens / meter ** 2
        g_K  = gbar_K*(n**4)              : siemens / meter ** 2
        """

        eqs_I = """
        I_leak = g_leak * (v - E_leak)   : amp / meter ** 2
        I_Na =   g_Na   * (v - E_Na)     : amp / meter ** 2
        I_K =    g_K    * (v - E_K)      : amp / meter ** 2
        I                                : amp / meter ** 2
        """

        # here you add the equations defining the potassium activation gates
        eqs_activation = """
        n_inf =  1/(1+exp((-53*mV - v)/(15*mV))) : 1
        m_inf =  1/(1+exp((-40*mV - v)/(9*mV)))  : 1
        h_inf =  1/(1+exp((-62*mV - v)/(-7*mV))) : 1
        taun  =  1.1*ms + 4.7*exp(-(-79*mV-v)**2/(50*mV)**2) *ms : second
        taum  =  .04*ms + .46*exp(-(-38*mV-v)**2/(30*mV)**2) *ms : second
        tauh  =  1.2*ms + 7.4*exp(-(-67*mV-v)**2/(20*mV)**2) *ms : second
        dm/dt = (m_inf - m)/taum : 1
        dh/dt = (h_inf - h)/tauh : 1
        dn/dt = (n_inf - n)/taun : 1
        """

        eqs_var = """
        E_leak  : volt
        E_Na    : volt
        E_K     : volt
        
        g_leak  : siemens / meter ** 2
        gbar_Na : siemens / meter ** 2
        gbar_K  : siemens / meter ** 2
        
        Cm      : farad / meter ** 2
        """

        eqs = eqs_V
        eqs += eqs_cond
        eqs += eqs_I
        eqs += eqs_activation
        eqs += eqs_var

        self.neurons = NeuronGroup(
            self.N, eqs, method="euler", dt=params["dt"], threshold="v>Vt", namespace={"Vt": params["Vt"]}, reset="")

        self.neurons.E_leak = params["E_leak"]
        self.neurons.E_Na = params["E_Na"]
        self.neurons.E_K = params["E_K"]
        self.neurons.g_leak = params["g_leak"]
        self.neurons.gbar_Na = params["gbar_Na"]
        self.neurons.gbar_K = params["gbar_K"]
        self.neurons.Cm = params["Cm"]

        self.neurons.m = 0.06452912
        self.neurons.h = 0.57323377
        self.neurons.n = 0.32350875
        self.neurons.v = -64.06540041*mV

        # TODO: improve/actually make work the synapse model
        sources, targets = self.mat.nonzero()
        eqs_synapse = """
        v_syn = w * mV : volt
        w              : 1
        """
        self.synapses = Synapses(
            self.neurons, self.neurons, eqs_synapse, on_pre="v += v_syn", dt=params["dt"])
        self.synapses.connect()
        self.synapses.w[:] = self.mat.flatten()

        self.statemon = StateMonitor(self.neurons, ["v", "m", "h", "n", "I",
                                                    "g_K", "g_Na", "I_leak", "I_Na", "I_K"], record=0)
        self.spikemon = SpikeMonitor(self.neurons)
        self.popmon = PopulationRateMonitor(self.neurons)
        self.net = Network([self.neurons, self.synapses,
                           self.statemon, self.spikemon, self.popmon])
