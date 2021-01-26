import numpy as np
import pandas as pd
from .standard_hhm import *


# fmt: off
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


class Channels:
    def __init__(
        self, transition_rates, open_state, initialization_weights, num_channels
    ):
        self._transitions = transition_rates
        self.open_state = open_state
        self.num_channels = num_channels
        self.states = Channels.draw_initial(initialization_weights, self.num_channels)

    @staticmethod
    def draw_initial(weights, num):
        draws = draw_weighted(weights.keys(), size=num, p=weights.values())
        unique, counts = np.unique(draws, return_counts=True)

        initial = {state: 0 for state in weights.keys()}
        for state, initial_num in zip(unique, counts):
            initial[state] += initial_num
        return initial

    def rates_state(self, state, V):
        return {t: rate(V) for t, rate in self._transitions.items() if t[0] == state}

    def rate_total(self, V):
        rate_total = 0
        for transition, rate in self._transitions.items():
            rate_total += self.states[transition[0]] * rate(V)
        return rate_total

    def open_proportion(self):
        return self.states[self.open_state] / self.num_channels

    def transition_weights(self, V):
        weights = {}
        for transition, rate in self._transitions.items():
            weights[transition] = self.states[transition[0]] * rate(V)
        return weights

    def make_transition(self, transition):
        self.states[transition[0]] -= 1
        self.states[transition[1]] += 1


def KChannels(num=100000):
    return Channels(automaton_K, 5, weights_K_0, num)


def NaChannels(num=100000):
    return Channels(automaton_Na, 4, weights_Na_0, num)


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


def stochastic_transition(channels, V, t_step):
    for state in channels.states.keys():
        rates_state = channels.rates_state(state, V)
        transitioning = np.random.binomial(
            channels.states[state], t_step * sum(rates_state.values())
        )
        channels.states[state] -= transitioning

        trans_rates = {s2: rate for (s1, s2), rate in rates_state.items()}
        for new_state in draw_weighted(
            trans_rates.keys(), size=transitioning, p=trans_rates.values()
        ):
            channels.states[new_state] += 1


def stochastic_vc_current(channels, V, E, g, t_step=0.01, t_max=30):
    ts = np.arange(0, t_max + t_step, t_step)
    ps = [channels.open_proportion()]

    for t in ts[:-1]:
        stochastic_transition(channels, V(t), t_step)
        ps.append(channels.open_proportion())

    return ts, g * np.array(ps) * (np.array([V(t) for t in ts]) - E)


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
        # print(last["open_rate_K"])
        # print(last["open_rate_Na"])

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
