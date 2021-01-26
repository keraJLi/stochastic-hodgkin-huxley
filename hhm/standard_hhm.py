import numpy as np
import pandas as pd

num_channels_total = 10000
E_Na = 50
E_K = -77
g_Na = 120
g_K = 36
g_leak = 0.3
E_leak = -54.387

rates = {
    "n": {
        "alpha": lambda V: 0.01 * (V + 55) / (1 - np.exp(-0.1 * (V + 55))),
        "beta": lambda V: 0.125 * np.exp(-0.0125 * (V + 65)),
    },
    "m": {
        "alpha": lambda V: 0.1 * (V + 40) / (1 - np.exp(-0.1 * (V + 40))),
        "beta": lambda V: 4 * np.exp(-0.0556 * (V + 65)),
    },
    "h": {
        "alpha": lambda V: 0.07 * np.exp(-0.05 * (V + 65)),
        "beta": lambda V: 1 / (1 + np.exp(-0.1 * (V + 35))),
    },
}

dx = {  # ugly late binding fix
    name_x: lambda x, V, rates_x=rates_x: rates_x["alpha"](V) * (1 - x)
    - rates_x["beta"](V) * x
    for name_x, rates_x in rates.items()
}


def I_leak(V):
    return g_leak * (V - E_leak)


def vc_current_K(
    V, n_0=0.3117, t_max=100, t_step=0.01,
):
    ts = np.arange(0, t_max + t_step, t_step)
    ns = [n_0]

    for t in ts[:-1]:
        ns.append(ns[-1] + t_step * dx["n"](ns[-1], V(t)))

    return ts, g_K * np.array(ns) ** 4 * (np.array([V(t) for t in ts]) - E_K)


def vc_current_Na(
    V, m_0=0.0529, h_0=0.5961, t_max=100, t_step=0.01,
):
    ts = np.arange(0, t_max + t_step, t_step)
    ms, hs = [m_0], [h_0]

    for t in ts[:-1]:
        ms.append(ms[-1] + t_step * dx["m"](ms[-1], V(t)))
        hs.append(hs[-1] + t_step * dx["h"](hs[-1], V(t)))

    return (
        ts,
        g_Na * np.array(ms) ** 3 * np.array(hs) * (np.array([V(t) for t in ts]) - E_Na),
    )


def hhm(
    I_e=lambda t: 0,
    C_m=1,
    V_0=-65,
    n_0=0.3177,
    m_0=0.0529,
    h_0=0.5961,
    t_max=50,
    t_step=0.01,
):
    data = [
        {
            "t": 0,
            **{"n": n_0, "m": m_0, "h": h_0},
            "I_K": g_K * n_0 ** 4 * (V_0 - E_K),
            "I_Na": g_Na * m_0 ** 3 * h_0 * (V_0 - E_Na),
            "I_leak": I_leak(V_0),
            "I_e": I_e(0),
            "V": V_0,
        }
    ]

    for t in np.arange(0, t_max + t_step, t_step)[1:]:
        last = data[-1]
        update = {
            "t": last["t"] + t_step,
            "I_K": g_K * last["n"] ** 4 * (last["V"] - E_K),
            "I_Na": g_Na * last["m"] ** 3 * last["h"] * (last["V"] - E_Na),
            "I_leak": I_leak(last["V"]),
            "I_e": I_e(t),
        }
        integrate = {
            **{x: dx[x](last[x], last["V"]) for x in ["n", "m", "h"]},
            "V": (-last["I_Na"] - last["I_K"] - last["I_leak"] + last["I_e"]) / C_m,
        }
        integrate = {k: data[-1][k] + t_step * v for k, v in integrate.items()}
        data.append({**update, **integrate})

    return pd.DataFrame(
        data, columns=["t", "n", "m", "h", "I_K", "I_Na", "I_leak", "I_e", "V"],
    )
