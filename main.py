import numpy as np
from scipy.stats import norm

# Paramètres du contrat et du marché
S0 = 100.0      # prix spot aujourd'hui
K = 100.0       # strike
r = 0.05        # taux sans risque
sigma = 0.20    # volatilité
T = 1.0         # maturité (en années)
N = 100000     # nombre de simulations Monte-Carlo
def mc_call_price(S0, K, r, sigma, T, N):
    rng = np.random.default_rng(42)
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
    prix_mc = mc_call_price(S0, K, r, sigma, T, N)
    prix_bs = bs_call_price(S0, K, r, sigma, T)

    print(f"Prix Monte-Carlo : {prix_mc:.4f}")
    print(f"Prix Black-Scholes : {prix_bs:.4f}")
    print(f"Écart absolu : {abs(prix_mc - prix_bs):.4f}")