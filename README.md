# Monte Carlo vs. Black-Scholes — European Call Option Pricing

A small quantitative finance project that prices a European call option two ways —
by Monte Carlo simulation and by the closed-form Black-Scholes formula — and verifies
that the numerical estimate converges to the analytical reference.

## Motivation

The point is not to price the option (a closed-form solution already exists) but to
**validate a numerical method against an exact benchmark** — a core reflex of model
validation. When a simulation and an analytical formula agree, both the implementation
and the underlying assumptions are consistent.

## Model

Under the risk-neutral measure, the underlying follows a geometric Brownian motion,
so its value at maturity $T$ is:

$$S_T = S_0 \exp\left[\left(r - \tfrac{\sigma^2}{2}\right)T + \sigma\sqrt{T}\,Z\right], \quad Z \sim \mathcal{N}(0,1)$$

The price of a European call with strike $K$ is the discounted expected payoff:

$$C = e^{-rT}\,\mathbb{E}\left[\max(S_T - K,\, 0)\right]$$

**Monte Carlo** estimates this expectation by averaging the payoff over $N$ simulated
paths. **Black-Scholes** provides the exact closed form:

$$C = S_0\,\Phi(d_1) - K e^{-rT}\Phi(d_2), \qquad d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

## Results

With $S_0 = K = 100$, $r = 5\%$, $\sigma = 20\%$, $T = 1$ year and $N = 100{,}000$ simulations:

| Method | Price |
|--------|-------|
| Monte Carlo | 10.4205 |
| Black-Scholes | 10.4506 |
| Absolute difference | 0.0300 |

The two methods agree to within ~0.3%. The residual gap is Monte Carlo sampling error,
which decreases as $1/\sqrt{N}$ — multiplying $N$ by 10 shrinks it by about $\sqrt{10} \approx 3.2$.

## Running the project

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync         # create the environment and install dependencies
uv run main.py  # run the pricing and comparison
```

## Stack

Python 3.12 · NumPy · SciPy · managed with uv