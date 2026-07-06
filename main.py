import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Paramètres du contrat et du marché
S0 = 100.0      # prix spot aujourd'hui
K = 100.0       # strike
r = 0.05        # taux sans risque
sigma = 0.20    # volatilité
T = 1.0         # maturité (en années)
N = 100000     # nombre de simulations Monte-Carlo
N_Values=[100, 500, 1000, 5000, 10000, 50000, 100000]
def mc_call_price(S0, K, r, sigma, T, N, rng):
    Z = rng.standard_normal(N)
    S_T = S0 * np.exp((r - sigma**2/2)*T + sigma*np.sqrt(T)*Z)
    payoffs = np.maximum(S_T - K,0)
    prix = np.exp(-r*T) * np.mean(payoffs)
    
    return prix

def bs_call_price(S0, K, r, sigma, T):
    d1 = (np.log(S0/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 -sigma*np.sqrt(T)
    prix= S0 * norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    return prix

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    prix_mc = mc_call_price(S0, K, r, sigma, T, N, rng)
    prix_bs = bs_call_price(S0, K, r, sigma, T)
    errors = []
    for N in N_Values:
         erreurs = [abs(mc_call_price(S0, K, r, sigma, T, N, rng) - prix_bs) for _ in range(50)]
         errors.append(np.mean(erreurs))

    print(f"Prix Monte-Carlo : {prix_mc:.4f}")
    print(f"Prix Black-Scholes : {prix_bs:.4f}")
    print(f"Écart absolu : {abs(prix_mc - prix_bs):.4f}")
    for N, err in zip(N_Values, errors): # zip associe les deux listes élément par élément — au premier tour tu obtiens le 1er N avec la 1re erreur, etc. C'est la façon idiomatique de parcourir deux listes en parallèle en Python.
        print(f"N = {N:>7d} → erreur moyenne = {err:.4f}")
        
    # Graphique de convergence
    plt.figure(figsize=(8, 5))
    plt.loglog(N_Values, errors, marker="o", label="Erreur Monte-Carlo")

    # Droite de référence théorique en 1/sqrt(N)
    ref = errors[0] * (N_Values[0] / np.array(N_Values))**0.5
    plt.loglog(N_Values, ref, linestyle="--", color="gray", label=r"Pente théorique $1/\sqrt{N}$")

    plt.xlabel("Nombre de simulations N")
    plt.ylabel("Erreur absolue moyenne")
    plt.title("Convergence Monte-Carlo vers Black-Scholes")
    plt.legend()
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.savefig("convergence.png", dpi=150, bbox_inches="tight")
    plt.show()