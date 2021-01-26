import numpy as np
import pandas as pd
from .stochastic_hhm import *


def draw_t_step(lambda_):
    r = np.random.random()
    return np.log(1 / r) / lambda_


def markov_channel_step(channelss, V, t, t_max):
    channels_weights = {channels: channels.rate_total(V) for channels in channelss}

    rate_total = sum(channels_weights.values())
    t_step = draw_t_step(rate_total)
    if t + t_step > t_max:
        t_step = t_max - t

    drawn_channels = draw_weighted(channels_weights.keys(), p=channels_weights.values())

    transitions = drawn_channels.transition_weights(V)
    drawn_transition = draw_weighted(transitions.keys(), p=transitions.values())
    drawn_channels.make_transition(drawn_transition)
    return t + t_step


def markov_vc_current(channels, V, E, g, t_max=100):
    ts = [0]
    ps = [channels.open_proportion()]

    while ts[-1] < t_max:
        new_t = channel_step([channels], V(ts[-1]), ts[-1], t_max)
        ts.append(new_t)
        ps.append(channels.open_proportion())

    return ts, g * np.array(ps) * (np.array([V(t) for t in ts]) - E)


def markov_hhm(
    I_e=lambda t: 0, C_m=1, num_K=1000, num_Na=1000, V_0=-65, t_max=50,
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

    last = data[0]
    while last["t"] < t_max:
        last = data[-1]
        new_t = markov_channel_step(
            [k_channels, na_channels], last["V"], last["t"], t_max
        )

        update = {
            "t": new_t,
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
        integrate = {
            k: data[-1][k] + (new_t - last["t"]) * v for k, v in integrate.items()
        }
        data.append({**update, **integrate})
        print(last["t"])

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
