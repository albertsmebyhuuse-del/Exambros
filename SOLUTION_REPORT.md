# MST 06412 solution report

The 30% to 40% capital tax reform lowers constant consumption from 1 to 0.922696. Accounting for the transition moves the revenue-maximizing tax on the specified grid from 70% to 80%. In the portfolio problem, the baseline actor’s maximum stock-share error is 0.004204, compared with a finite-difference hedge as large as 0.141194. At the mean state of the reduced pricing model, the firm switches from local-currency to producer-currency pricing at a marginal-cost exposure of about 0.8708 under Kimball demand and 0.4686 under CES demand.

[Executed solution notebook](ExamMST0641_2026_Solution.ipynb)

## 1. A capital tax in an open economy

### (a) Steady states and static revenue

I solved the utilization condition for $u$ and used $i/k=z+(q-1)/\eta$ to express the dynamics in $k$ and $q$. The two residuals given to `fsolve` were investment minus depreciation per unit of capital and the costate drift. The old numerical steady state agreed with the closed form to $2.7\times10^{-15}$. Using it as the starting guess at the new tax gave the following states.

| Tax | Capital $k^*$ | Tobin’s $q^*$ | Utilization $u^*$ |
|---|---:|---:|---:|
| 30% | 5.915722 | 1.000000 | 1.000000 |
| 40% | 5.151528 | 0.968985 | 0.966111 |

For the static curve I increased the tax in five-point increments, initialized each solve with the previous root, and evaluated revenue net of the depreciation allowance at the new root. The largest steady-state revenue was **0.313837 at 70%**.

| Tax | Cold-start `ier` | Returned $k$ | Continued $k^*$ | Cold residual max norm |
|---|---:|---:|---:|---:|
| 70% | 5 | 5.915722 | 2.574530 | 0.04196 |
| 75% | 5 | 5.915722 | 2.102737 | 0.04623 |
| 90% | 5 | 5.915722 | 0.673498 | 0.05710 |

Each cold solve returned the initial capital and flagged a lack of progress. The residuals are far from zero. Continuation keeps the initial guess close to the positive steady-state branch as the tax rises.

### (b) Shooting the transition

Capital starts at 5.915722 and is predetermined. I bisected the initial value of $q$ between 0.3 and 1. For each candidate, a downward crossing of the new steady-state capital marked a low shot. An upward crossing of zero capital growth marked a high shot. Stopping at the first event avoids integrating a missed trajectory far along the unstable direction.

The initial value is $q(0)=0.896271119352$, and the bisected trajectory exits at year 209.095. Utilization jumps from 1 to 0.960583 and then rises toward 0.966111. Capital falls along the transition. Tobin’s $q$ and the investment rate jump below their new steady-state values and rise toward them.

### (c) Consumption and foreign assets

Discounting the foreign-asset equation gives $d(e^{-rt}b)/dt=e^{-rt}(n-c)$, where $n$ is household income net of investment and adjustment costs. With zero initial assets and the discounted terminal condition, constant consumption equals the annuity value of net income. I integrated discounted net income on 20,001 equally spaced points along the shot and added the discounted constant steady-state tail.

| Method | Consumption |
|---|---:|
| Shooting and trapezoid quadrature | 0.922695980200 |
| Joint boundary-value solve | 0.922695965833 |

The second method solves for $k,q,b$ and the unknown constant $c$ jointly, with terminal capital at its steady state and zero terminal asset growth at year 200. The methods differ by $1.44\times10^{-8}$ in consumption and $1.81\times10^{-13}$ in initial $q$. The forward current-account integration shows foreign assets growing from zero to 0.973203 at year 100 as the household accumulates assets to support its constant consumption.

### (d) Dynamic revenue

I reused the continued steady states as shooting targets and applied the same discounted quadrature to tax receipts. At 30% the economy stays in its original steady state, so the annuity equals stationary revenue. At 40%, the annuity is 0.231048, a gain of 0.051626 over 30%. The transition leaves more capital available to tax for part of the horizon, raising the maximum annuity to **0.360986 at 80%**.

## 2. Dynamic portfolio choice

### (a) Closed-form benchmarks

With zero return–state correlation, the hedging term vanishes and the stock share is $X/(\gamma\sigma)$. As risk aversion tends to one, the log-utility consumption share is $\rho$ and the stock share is $X/\sigma$. I plotted both shares over three stationary standard deviations on either side of the mean.

### (b) Finite-difference policy iteration

For a fixed consumption and stock policy, I used the drift sign to select the forward or backward first difference and used a centered second difference. Equal endpoint and ghost values implement reflection, so the missing neighbor coefficient is folded into the corresponding diagonal. The resulting tridiagonal system is solved with `spsolve`. The first-order conditions then update consumption from $\phi^{-1/\gamma}$ and the stock share from the sum of the myopic share and a term proportional to $\phi'/\phi$. The maximum relative change in the value coefficient fell below $10^{-10}$ after **four improvements**.

At $\gamma=1.0001$, the largest absolute deviations from the log benchmarks were $5.93\times10^{-6}$ in consumption share and $3.48\times10^{-4}$ in stock share. Widening the grid from six to eight stationary standard deviations changes the stock share by at most $1.25\times10^{-5}$ in the reporting region. Using only three standard deviations changes it by up to 0.13333 because the reflecting boundaries then sit at the reporting endpoints and affect the value slope used in the hedge.

### (c) Critic and actor networks

Both networks take the standardized Sharpe ratio. Exponentiating the critic output makes $\phi$ positive, and exponentiating the actor’s log-consumption output makes consumption positive. Automatic differentiation gives $\phi'$ and $\phi''$ with respect to the original state $X$. On each uniform batch I held the critic fixed and maximized the mean normalized HJB residual over the actor, then held the updated actor fixed and minimized the mean squared normalized residual over the critic. Each run used seed 0, 5,000 pairs of Adam steps, batch size 256, and learning rates decaying from $10^{-3}$ to $5\times10^{-5}$.

| Baseline network measure on the reporting grid | Result |
|---|---:|
| Maximum relative value-coefficient error | $6.60\times10^{-4}$ |
| Maximum absolute consumption-share error | $2.31\times10^{-5}$ |
| Maximum absolute stock-share error | $4.20\times10^{-3}$ |
| $\log_{10}$ RMSE of the normalized HJB residual | −4.2695 |
| Maximum absolute finite-difference hedge | 0.141194 |

### (d) Hedging across parameters

I subtracted the myopic policy from each stock policy and compared the two methods for all six risk-aversion and correlation combinations. At zero correlation the finite-difference hedge is zero by construction. The actor’s deviations from zero in those panels measure policy approximation error. For the baseline case, narrowing the training window to three stationary standard deviations raises the maximum stock-share error from 0.00420 to 0.02499. The error plot shows how the two training windows compare across the state range.

## 3. Currency choice with menu costs

### (a) Desired price and sector-price interpolation

For every exogenous state I maximized current LCP profit over the posted-price grid. The implied desired price is stepwise because prices lie on a grid. Reconstructing next period’s sector price from its two neighboring nodes reproduced the transition to within $1.67\times10^{-16}$ for interior moves.

| Sector-price path | Fitted desired pass-through | Flexible-price approximation |
|---|---:|---:|
| Fixed at the mean | 0.427311 | 0.428571 |
| Moves with the exchange rate | 0.646305 | 0.642857 |

### (b) Conditional expectation

I multiplied along the productivity axis, gathered the two sector-price nodes for each possible next exchange rate and weighted them, then multiplied along the exchange-rate axis. The sector-price gather needs the index of the next exchange-rate shock, so it precedes that final average. A direct sum on the small grid agreed with the vectorized result to $3.33\times10^{-16}$. A dense transition over the full exogenous state would require **1,216,265,625 float64 entries**, or 9.06 GiB.

### (c) Value iteration and reset policies

For each future state, I compared the value of keeping the existing contract with the best value across both currencies and all reset prices after subtracting the menu cost. I used the part (c) starting values $\Pi_c/(1-\delta)$. The Bellman operator preserves order and discounts an additive constant by $\delta$, which yields the MacQueen–Porteus bounds. Once their width fell below $10^{-6}$, I used the midpoint correction. The reduced grid took **980 iterations** and the full grid **924 iterations**. The notebook records run times for both and plots the two dollar reset prices and the LCP inaction band. Ties at a reset are assigned to LCP, and ties between keeping and resetting retain the existing contract.

### (d) Indifference threshold

I used Brent’s method on the difference between the maximized LCP and PCP values at the reduced model’s mean state. Each function evaluation resolves the Bellman equation.

| Demand | Indifference exposure $\phi^*$ | Exposure for flexible pass-through $1/2$ |
|---|---:|---:|
| Kimball | 0.870763 | 0.875000 |
| CES | 0.468559 | 0.500000 |

The reduced-model currency maps display the choices below and above the Kimball root across productivity and exchange-rate states.

### (e) Simulated price changes

I simulated independent productivity and exchange-rate transitions for each of 5,000 firms over 600 months. Each firm draws its sector-price node using the interpolation weights and follows the keep/reset policy at the new state. I discarded the first 120 months. A price change counts when the firm resets. To measure its size, I valued the old contract and the new one at the same month’s exchange rate. Spell durations use consecutive resets that both occur after the burn-in. Histograms group resets by the currency chosen at the reset.

| Full-model simulation measure | Result |
|---|---:|
| Monthly reset frequency | 14.4930% |
| Median completed spell | 5 months |
| Mean absolute log dollar-price change at a reset | 0.115567 |
| LCP share of firm-months | 89.3996% |

The seed-0 panel contained an LCP firm that kept its currency through months 120–240 but no PCP firm that did so. For the PCP trace I used firm 1385 from an independent full-model simulation with seed 134 and the same parameters. The summary statistics and histograms use seed 0.

### (f) Pass-through between resets

For each firm, I paired adjacent post-burn-in resets and kept pairs with the same currency after both dates. I regressed the difference in the dollar prices set at those dates on the exchange-rate change between them, with an intercept. This measures a change across two reset dates, while part (e) measures the repricing at one date.

| Marginal-cost exposure | Mean-state choice | LCP share | LCP slope (pairs) | PCP slope (pairs) |
|---|---|---:|---:|---:|
| 0.720763 | LCP | 92.9665% | 0.429311 (302,233) | 0.911147 (2,560) |
| 1.020763 | PCP | 3.2625% | 1.498668 (3,191) | 0.523535 (315,334) |

The currency preferred at the mean state also dominates simulated firm-months at each exposure. Each minority-currency slope uses a much smaller set of same-currency pairs. These slopes describe the selected simulated reset observations.

## Reproducing the notebook

Open the notebook in Jupyter with NumPy, SciPy, Matplotlib and PyTorch installed, then restart the kernel and run all cells. The saved run used Python 3.12, NumPy 2.3.5, SciPy 1.17.0, Matplotlib 3.10.8 and PyTorch 2.15.0 on CPU. Training all seven network configurations and solving the full pricing model account for most of the run time.
