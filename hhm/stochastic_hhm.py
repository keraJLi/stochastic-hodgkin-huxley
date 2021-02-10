import numpy as np
import pandas as pd
from .standard_hhm import *


# fmt: off
# initialization weights of ion channel states. We have measured the steady 
# state states of K channels, which have a distribution approximately like
# the one on the bottom
weights_K_0 = {
    1: 20,
    2: 40,
    3: 30,
    4: 9,
    5: 1,
}

weights_Na_0 = {
    1: 50,
    2: 9,
    3: 1,
    4: 0,
    5: 40,
}

# Transitions and corresponding rates
automaton_K = {
    (1, 2): lambda V: 4 * rates["n"]["alpha"](V),
    (2, 1): lambda V: rates["n"]["beta"](V), 
    (2, 3): lambda V: 3 * rates["n"]["alpha"](V),
    (3, 2): lambda V: 2 * rates["n"]["beta"](V),
    (3, 4): lambda V: 2 * rates["n"]["alpha"](V),
    (4, 3): lambda V: 3 * rates["n"]["beta"](V), 
    (4, 5): lambda V: rates["n"]["alpha"](V),
    (5, 4): lambda V: 4 * rates["n"]["beta"](V),
}

automaton_Na = {
    (1, 2): lambda V: 3 * rates["m"]["alpha"](V),
    (2, 1): lambda V: rates["m"]["beta"](V),
    (2, 3): lambda V: 2 * rates["m"]["alpha"](V),
    (2, 5): lambda V: 0.24,  # k1
    (3, 2): lambda V: 2 * rates["m"]["beta"](V),
    (3, 4): lambda V: rates["m"]["alpha"](V),
    (3, 5): lambda V: 0.4,  # k2
    (4, 3): lambda V: 3 * rates["m"]["beta"](V), 
    (4, 5): lambda V: 1.5,  # k3
    (5, 3): lambda V: rates["h"]["alpha"](V),
}
# fmt: on


# Class representing the set of a type of ion channel within the membrane
class Channels:
    def __init__(
        self, transition_rates, open_state, initialization_weights, num_channels
    ):
        self._transitions = transition_rates
        self.open_state = open_state
        self.num_channels = num_channels
        self.states = Channels.draw_initial(initialization_weights, self.num_channels)

    # draws initial states of channels according to the given weights
    @staticmethod
    def draw_initial(weights, num):
        draws = draw_weighted(weights.keys(), size=num, p=weights.values())
        unique, counts = np.unique(draws, return_counts=True)

        initial = {state: 0 for state in weights.keys()}
        for state, initial_num in zip(unique, counts):
            initial[state] += initial_num
        return initial

    # returns the transitions and rates from one state to all others
    def rates_state(self, state, V):
        return {t: rate(V) for t, rate in self._transitions.items() if t[0] == state}

    # returns the complete transitioning rate (sum of all transitions)
    def rate_total(self, V):
        rate_total = 0
        for transition, rate in self._transitions.items():
            rate_total += self.states[transition[0]] * rate(V)
        return rate_total

    # returns the proportion of open channels (I didnt call it rate to avoid confusion
    # with transition rates)
    def open_proportion(self):
        return self.states[self.open_state] / self.num_channels

    # returns transition "weights" of all transitions meaning number of states * rate
    def transition_weights(self, V):
        weights = {}
        for transition, rate in self._transitions.items():
            weights[transition] = self.states[transition[0]] * rate(V)
        return weights

    # auxiliary
    def make_transition(self, transition):
        self.states[transition[0]] -= 1
        self.states[transition[1]] += 1


# easy initialization methods of K+ and Na+ channels
def KChannels(num=100000):
    return Channels(automaton_K, 5, weights_K_0, num)


def NaChannels(num=100000):
    return Channels(automaton_Na, 4, weights_Na_0, num)


# helper method to draw from a set given weights (uses weight / sum weights as pmf)
def draw_weighted(a, size=None, replace=True, p=None):
    a = list(a)
    p = np.fromiter(p, dtype=float)
    choice = np.random.choice(
        np.arange(len(a)), size=size, replace=replace, p=p / np.sum(p)
    )
    if size is None:
        return a[choice]
    else:
        return [a[c] for c in choice]


# applies the stochastic state transitions within a timeframe of t_step
def stochastic_transition(channels, V, t_step):
    for state in channels.states.keys():
        # we can draw the number of channels transitioning from a certain
        # state by a binomial distribution, because the transition of only
        # one channel is bernoulli distributed around t_step * sum rates
        rates_state = channels.rates_state(state, V)
        transitioning = np.random.binomial(
            channels.states[state], min(1, t_step * sum(rates_state.values()))
        )
        channels.states[state] -= transitioning

        # now for as many channels as are transitioning we draw a new state
        # weighted according to the transition rate to that new state
        trans_rates = {s2: rate for (s1, s2), rate in rates_state.items()}
        for new_state in draw_weighted(
            trans_rates.keys(), size=transitioning, p=trans_rates.values()
        ):
            channels.states[new_state] += 1


# stochastic current of a membrane with only one type of channel under voltage clamp
def stochastic_vc_current(channels, V, E, g, t_step=0.01, t_max=30):
    ts = np.arange(0, t_max + t_step, t_step)
    ps = [channels.open_proportion()]

    for t in ts[:-1]:
        stochastic_transition(channels, V(t), t_step)
        ps.append(channels.open_proportion())

    return ts, g * np.array(ps) * (np.array([V(t) for t in ts]) - E)


# stochastic simulation of the hodgkin huxley model with a current injection dependend on t
def stochastic_hhm(
    I_e=lambda t: 0, C_m=1, num_K=1000, num_Na=1000, V_0=-65, t_step=0.01, t_max=50,
):
    k_channels = KChannels(num_K)
    na_channels = NaChannels(num_Na)

    data = [
        {
            "t": 0,
            "open_rate_K": k_channels.open_proportion(),
            "open_rate_Na": na_channels.open_proportion(),
            "I_K": g_K * k_channels.open_proportion() * (V_0 - E_K),
            "I_Na": g_Na * na_channels.open_proportion() * (V_0 - E_Na),
            "I_leak": I_leak(V_0),
            "I_e": I_e(0),
            "V": V_0,
        }
    ]

    for t in np.arange(0, t_max + t_step, t_step):
        last = data[-1]
        stochastic_transition(k_channels, last["V"], t_step)
        stochastic_transition(na_channels, last["V"], t_step)

        update = {
            "t": last["t"] + t_step,
            "open_rate_K": k_channels.open_proportion(),
            "open_rate_Na": na_channels.open_proportion(),
            "I_K": g_K * last["open_rate_K"] * (last["V"] - E_K),
            "I_Na": g_Na * last["open_rate_Na"] * (last["V"] - E_Na),
            "I_leak": I_leak(last["V"]),
            "I_e": I_e(last["t"]),
        }
        integrate = {
            "V": (-last["I_Na"] - last["I_K"] - last["I_leak"] + last["I_e"]) / C_m
        }
        integrate = {k: data[-1][k] + t_step * v for k, v in integrate.items()}
        data.append({**update, **integrate})

    return pd.DataFrame(
        data,
        columns=[
            "t",
            "I_K",
            "I_Na",
            "I_leak",
            "I_e",
            "V",
            "open_rate_K",
            "open_rate_Na",
        ],
    )
