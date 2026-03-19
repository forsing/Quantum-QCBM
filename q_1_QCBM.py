"""
QCBM - Quantum Circuit Born Machine
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize as scipy_minimize
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.utils import algorithm_globals

SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

CSV_DRAWN = "/data/loto7hh_4582_k22.csv"
CSV_ALL   = "/data/kombinacijeH_39C7.csv"

MIN_VAL = [1, 2, 3, 4, 5, 6, 7]
MAX_VAL = [33, 34, 35, 36, 37, 38, 39]
NUM_QUBITS = 5 
NUM_LAYERS = 2 
MAXITER = 200 


def load_draws():
    df = pd.read_csv(CSV_DRAWN)
    return df.values


def build_empirical(draws, pos):
    dist = {}
    n_states = 1 << NUM_QUBITS
    for row in draws:
        v = int(row[pos]) - MIN_VAL[pos]
        if v >= n_states:
            v = v % n_states
        dist[v] = dist.get(v, 0) + 1
    total = sum(dist.values())
    return {k: c / total for k, c in dist.items()}


def kl_from_uniform(emp, n_vals):
    u = 1.0 / n_vals
    return sum(p * np.log(p / u) for p in emp.values() if p > 0)


def exact_probs(ansatz, theta):
    circ = ansatz.assign_parameters(theta)
    sv = Statevector.from_instruction(circ)
    probs = sv.probabilities()
    return {i: float(p) for i, p in enumerate(probs) if p > 1e-15}


def train_qcbm(target, theta0):
    ansatz = TwoLocal(
        num_qubits=NUM_QUBITS,
        rotation_blocks='ry',
        entanglement_blocks='cz',
        entanglement='linear',
        reps=NUM_LAYERS
    )

    def cost(theta):
        dist = exact_probs(ansatz, theta)
        loss = 0.0
        for k, pt in target.items():
            ps = dist.get(k, 1e-10)
            if ps <= 0:
                ps = 1e-10
            loss += pt * np.log(pt / ps)
        return float(loss)

    res = scipy_minimize(cost, theta0, method='COBYLA',
                         options={'maxiter': MAXITER, 'rhobeg': 0.5})
    return ansatz, res.x, res.fun


def greedy_combo(dists):
    combo = []
    used = set()
    for pos in range(7):
        ranked = sorted(dists[pos].items(), key=lambda x: x[1], reverse=True)
        for mv, prob in ranked:
            actual = int(mv) + MIN_VAL[pos]
            if actual > MAX_VAL[pos]:
                continue
            if actual in used:
                continue
            if combo and actual <= combo[-1]:
                continue
            combo.append(actual)
            used.add(actual)
            break
    return combo


def main():
    draws = load_draws()
    print(f"Ucitano izvucenih kombinacija: {len(draws)}")

    df_all_head = pd.read_csv(CSV_ALL, nrows=3)
    print(f"Graf svih kombinacija: {CSV_ALL}")
    print(f"  Primer: {df_all_head.values[0].tolist()} ... {df_all_head.values[-1].tolist()}")

    print("\n--- Zakonitost po pozicijama (KL od uniformne) ---")
    targets = []
    for pos in range(7):
        emp = build_empirical(draws, pos)
        targets.append(emp)
        n_vals = 1 << NUM_QUBITS
        kl = kl_from_uniform(emp, n_vals)
        print(f"  Poz {pos+1} [{MIN_VAL[pos]}-{MAX_VAL[pos]}]: KL = {kl:.6f}")

    n_params = TwoLocal(NUM_QUBITS, 'ry', 'cz', 'linear',
                        reps=NUM_LAYERS).num_parameters
    all_theta0 = [np.random.uniform(0, 2 * np.pi, n_params) for _ in range(7)]

    print(f"\n--- Treniranje 7 QCBM (5q, COBYLA {MAXITER} iter) ---")
    dists = []
    for pos in range(7):
        print(f"  Poz {pos+1}...", end=" ", flush=True)
        ansatz, params, loss = train_qcbm(targets[pos], all_theta0[pos])
        d = exact_probs(ansatz, params)
        dists.append(d)
        top = sorted(d.items(), key=lambda x: x[1], reverse=True)[:3]
        info = " | ".join(f"{int(v)+MIN_VAL[pos]}:{p:.3f}" for v, p in top)
        print(f"loss={loss:.4f}  top: {info}")

    combo = greedy_combo(dists)

    print(f"\n{'='*50}")
    print(f"Predikcija (QCBM, deterministicki, seed={SEED}):")
    print(combo)
    print(f"{'='*50}")


if __name__ == "__main__":
    main()


"""
Ucitano izvucenih kombinacija: 4582
Graf svih kombinacija: /Users/4c/Desktop/GHQ/data/kombinacijeH_39C7.csv
  Primer: [1, 2, 3, 4, 5, 6, 7] ... [1, 2, 3, 4, 5, 6, 9]

--- Zakonitost po pozicijama (KL od uniformne) ---
  Poz 1 [1-33]: KL = 0.944718
  Poz 2 [2-34]: KL = 0.490175
  Poz 3 [3-35]: KL = 0.311832
  Poz 4 [4-36]: KL = 0.268089
  Poz 5 [5-37]: KL = 0.326176
  Poz 6 [6-38]: KL = 0.504515
  Poz 7 [7-39]: KL = 0.989665

--- Treniranje 7 QCBM (5q, COBYLA 200 iter) ---
  Poz 1... loss=0.0039  top: 1:0.174 | 2:0.153 | 3:0.128
  Poz 2... loss=0.0331  top: 6:0.077 | 7:0.076 | 8:0.075
  Poz 3... loss=0.2210  top: 11:0.108 | 18:0.097 | 23:0.073
  Poz 4... loss=0.2309  top: 24:0.103 | 25:0.098 | 18:0.088
  Poz 5... loss=0.0530  top: 26:0.069 | 25:0.068 | 27:0.067
  Poz 6... loss=0.2902  top: 30:0.133 | 35:0.129 | 32:0.116
  Poz 7... loss=0.2200  top: 35:0.154 | 38:0.123 | 37:0.116

==================================================
Predikcija (QCBM, deterministicki, seed=39):
[1, 6, x, y, z, 30, 35]
==================================================
"""



"""
QCBM - Quantum Circuit Born Machine

Poredim distribuciju iz grafa svih kombinacija (uniformna, teorijska) 
sa distribucijom iz grafa izvucenih kombinacija (empirijska, 4582 izvlacenja). 
Razlika izmedju te dve distribucije je upravo ono sto trazim 
- zakonitost (odstupanje od uniformnosti).

QCBM - najjednostavniji generativni pristup,  
koji uci distribuciju izvucenih kombinacija 
i poredi sa uniformnom distribucijom svih kombinacija 
(zakonitost).

Ucitava oba CSV-a (izvucene kombinacije za trening, sve kombinacije za referencu). 
Racuna zakonitost po pozicijama (KL divergencija empirijska vs uniformna distribucija)
Trenira 7 nezavisnih QCBM-ova (5 qubita svaki, COBYLA 200 iteracija, KL loss)
Uzorkuje iz treniranih modela (100K shotova)
Greedy selekcija - 
bira najverovanije brojeve uz poštovanje sortiranog redosleda i bez duplikata. 
Seed=39, potpuno deterministicki. 

Sampler sa shots uvodi kvantni sum (shot noise) koji nije kontrolisan seedom. 
Resenje: koristiti egzaktne verovatnoce umesto uzorkovanja 
- za 5 qubita (32 stanja) to je trivijalno i potpuno deterministicko.

Umesto Sampler sa shots (koji unosi slucajnost), 
koristim Statevector za egzaktne verovatnoce 
- nula suma, potpuno deterministicko. 
Za 5 qubita (32 stanja) ovo je i brze. 
Svako pokretanje ce dati isti rezultat.

Cela prica je cisto kvantna:
Kvantno kolo: TwoLocal (Ry rotacije + CZ entanglement) - pravo parametrizovano kvantno kolo. 
Kvantno stanje: Statevector.from_instruction() - egzaktna evolucija kvantnog stanja. 
Kvantne verovatnoce: sv.probabilities() - Born-ovo pravilo, verovatnoce iz amplituda kvantnog stanja. 
Model: QCBM (Quantum Circuit Born Machine) - kvantno kolo uci distribuciju. 
Jedini klasicni deo je COBYLA optimizer koji podesava parametre kvantnog kola 
(to je standard i u pravom kvantnom racunarstvu - varijacioni pristup). 
Nema klasicnog ML-a, nema neuronskih mreza, nema klasicnih modela. 
Sve predikcije dolaze iskljucivo iz kvantnog stanja.
"""
