# Methodology

All rates are decimals and all times are year fractions. Prices are dirty prices at issuance, immediately before any cash flows. Synthetic inputs are generated locally with fixed seeds. No market data is downloaded at runtime.

## Discount curve and idealized OIS swaps

The curve stores $P_0(t)$, with $P_0(0)=1$. Between knots, $\log P_0(t)$ is linear. Consequently the instantaneous forward $f_0(t)=-\partial_t\log P_0(t)$ is constant on each interval and right-continuous at knots. Beyond the final knot the final forward is held constant. Continuous zero rates are $z(t)=-\log P_0(t)/t$.

For contiguous annual fixed-pay par OIS quotes $S_n$, compounded floating payments telescope to $1-P_0(n)$. With unit annual accruals and no payment lag:

$$S_n\sum_{i=1}^n P_0(i)=1-P_0(n),\qquad
P_0(n)=\frac{1-S_n\sum_{i=1}^{n-1}P_0(i)}{1+S_n}.$$

This is a single-curve, annual-period abstraction of SOFR OIS. It does not implement daily compounding conventions, calendars, day counts, settlement lags, stubs or basis curves. User CSV input has positive `time` and decimal, continuously compounded `zero_rate` columns. Negative zero rates are supported if discount factors remain positive.

## Hull-White Gaussian structure

The model is $dr=(\theta(t)-ar)dt+\sigma dW$, represented as $r=x+\phi$ with $dx=-axdt+\sigma dW$ and $x(0)=0$. Define:

$$B(t,T)=\frac{1-e^{-a(T-t)}}a,\quad
v(t)=\frac{\sigma^2}{2a}(1-e^{-2at}),\quad
q(t)=\frac{\sigma^2}{2a^2}(1-e^{-at})^2.$$

The deterministic shift is $\phi(t)=f_0(t)+q(t)$. Away from forward knots:

$$\theta(t)=f'_0(t)+af_0(t)+\frac{\sigma^2}{2a}(1-e^{-2at}).$$

Here $f'_0(t)=0$ within an interpolation interval. At knots the model's deterministic short-rate component jumps with the forward. `theta` returns only the regular drift, not a classical finite drift at those jumps. The Gaussian representation and integrated shift handle these jumps exactly. A smooth classical short-rate diffusion would require a smooth forward interpolator, which this library deliberately does not supply.

Conditional bond prices are:

$$P(t,T;r)=\frac{P_0(T)}{P_0(t)}
\exp\left[-B(t,T)(r-f_0(t))-\tfrac12 v(t)B(t,T)^2\right].$$

Writing $r=\phi(t)+x$ gives the equivalent exponent $-B(x+q)-vB^2/2$. At time zero this reproduces every discount factor exactly.

Let $I_t=\int_0^t x(u)du$. Its variance and its covariance with $x(t)$ are:

$$V_I(t)=\frac{\sigma^2}{a^2}\left[t-\frac{2(1-e^{-at})}{a}
+\frac{1-e^{-2at}}{2a}\right],\qquad
\operatorname{Cov}(x(t),I_t)=q(t).$$

The deterministic shift integrates to:

$$\int_s^t\phi(u)du=\log\frac{P_0(s)}{P_0(t)}
+\tfrac12\left[V_I(t)-V_I(s)\right].$$

A small-argument series avoids cancellation in $V_I$. This identity supplies exact integrated discounting without numerical integration of a discontinuous forward curve.

## European closed forms

For an option expiring at $E$ on a zero-coupon bond maturing at $T$, define $s_P=B(E,T)\sqrt{v(E)}$ and:

$$d_1=\frac{\log[P_0(T)/(KP_0(E))]}{s_P}+\frac{s_P}{2},\qquad d_2=d_1-s_P.$$

The call price is $P_0(T)N(d_1)-KP_0(E)N(d_2)$ and the put follows put-call parity. Zero volatility or zero expiry uses discounted intrinsic value. $N$ is implemented with `math.erf`.

For strictly positive bond cash flows $c_i$ at $T_i>E$, solve the unique root $r^*$ of $\sum_i c_iP(E,T_i;r^*)=K$. Jamshidian decomposition then prices the coupon-bond option as $\sum_i c_i\operatorname{Option}(E,T_i,P(E,T_i;r^*))$. Cash flows at or before expiry are excluded. Mixed-sign coupons are rejected because the required monotonic decomposition is not guaranteed.

A payer swaption on an annual fixed-pay swap is a put with strike one on a coupon bond with coupons equal to the swap strike and final principal one. A receiver swaption is the corresponding call. The library requires positive fixed coupons for this decomposition.

A simple-rate caplet fixing at $S$, paid at $T$, with $\delta=T-S$, is $(1+\delta K)$ times a put on $P(S,T)$ with strike $1/(1+\delta K)$. A floorlet uses a call. These are simple forward-rate instruments, not in-arrears compounded SOFR caps.

For normal volatility $\nu$, annuity $A$, forward $F$ and swap strike $K$, the Bachelier payer value is:

$$A[(F-K)N(d)+\nu\sqrt E\,n(d)],\qquad d=\frac{F-K}{\nu\sqrt E}.$$

At the money this is $A\nu\sqrt E/\sqrt{2\pi}$. The synthetic quote grid takes reference-model ATM prices, converts them to ATM normal vols, and applies seeded multiplicative noise. This is an internally generated calibration exercise, not independent market validation. Positive $a,\sigma$ are calibrated in log coordinates by a deterministic Nelder-Mead optimizer. The objective is mean squared relative price residual. Bounds prevent implausible search excursions and nonfinite arithmetic. The noiseless round-trip is separate from the noisy fit.

## Trinomial tree and curve fitting

The tree uses the Hull-White Arrow-Debreu alpha-fitting principle with an exact OU moment variant. It is not a literal reproduction of the original Euler-moment branching boundaries. For interval $h$, let $v_h=v(h)$ and $\Delta x=\sqrt{3v_h}$. At state $x_j=j\Delta x$, the conditional mean is $x_je^{-ah}$. Choose the nearest integer center $k$ and residual $e=je^{-ah}-k$. The three next states are $(k-1,k,k+1)\Delta x$, with:

$$p_d=\tfrac12(\tfrac13+e^2-e),\quad
p_m=\tfrac23-e^2,\quad
p_u=\tfrac12(\tfrac13+e^2+e).$$

Nearest-center residuals lie within half a grid spacing, which keeps probabilities nonnegative. These probabilities preserve the exact conditional mean and variance. The topology recombines and shifts centers inward as mean reversion increases. The zero-volatility case is deterministic.

Given discounted state prices $Q_{n,j}$, calibrate the shift for each interval:

$$\alpha_n=\frac1h\log\left[\frac{\sum_jQ_{n,j}e^{-x_jh}}{P_0(t_{n+1})}\right].$$

Propagate $Q$ using $e^{-(x_j+\alpha_n)h}p_{j,k}$. Summing state prices reproduces the next input discount factor. Pricing uses backward discounted expectations. Curve fitting is exact on the tree grid; option prices retain finite-step distribution and payoff discretization error. Convergence can oscillate as the strike moves relative to nodes. Tree dates must align exactly with the uniform grid. Misaligned events are rejected, not rounded.

Bond exercise occurs after payment of the current coupon: cap continuation at issuer call redemption or floor it at investor put redemption, then add the current coupon. Simultaneous call and put rights are rejected because no priority convention is specified. European and Bermudan options compare continuation with remaining-bond intrinsic value on permitted exercise dates. Underlying coupons can fall between exercise dates; the remaining-bond value excludes coupons already paid and the current coupon when exercise occurs on a payment date.

## Spreads and risk

OAS adds a constant $s$ only to discounting. It changes optimal exercise but does not alter risk-neutral state transitions, curve fitting or mortgage incentives. A safeguarded Brent solver finds $V(s)=V_{\mathrm{market}}$ on a specified bracket. An unbracketed target fails explicitly. Z-spread uses the same continuous spread on promised cash flows and ignores exercise.

For zero-curve parallel shifts $\pm b$, rebuild the tree or mortgage engine and retain the solved OAS and model parameters. Reevaluate exercise or prepayments:

$$D_{\mathrm{eff}}=\frac{V_- - V_+}{2V_0b},\qquad
C_{\mathrm{eff}}=\frac{V_-+V_+-2V_0}{V_0b^2}.$$

Risk bumps do not recalibrate $a,\sigma$ and do not solve a fresh OAS. Mortgage bumps reuse the random seed. Key-rate bumps use piecewise-linear hat weights at the curve knots, with constant end weights. The weights partition a parallel shift. Sum-of-key-rate duration reconciles with parallel duration in the small-bump limit, up to finite-difference and exercise-boundary effects.

## Mortgage cash flows and prepayments

Gross monthly note rate is $i=c/12$. For opening surviving balance $B$ and remaining months $n$, scheduled payment is $Bi/[1-(1+i)^{-n}]$, scheduled principal is payment minus $Bi$, and the zero-rate limit is $B/n$. Pool prepayments remove whole loans proportionally. Surviving loans retain their amortization term. Recomputing payment from surviving balance reflects fewer identical loans, not individual borrower payment recasting.

For CPR, monthly mortality is $\mathrm{SMM}=1-(1-\mathrm{CPR})^{1/12}$. Prepaid principal is SMM times the balance after scheduled principal. Final-month principal is forced to the remaining balance. Investor interest is $B(c-f_s-f_g)/12$, stripping servicing and guarantee fees. The examples use stylized fee assumptions, not agency fee schedules. There are no default losses or remittance delays.

The PSA baseline is $\mathrm{CPR}_m=\min[0.002\min(m,30)\cdot\mathrm{speed}/100,1]$, where speed is a percentage of PSA and $m$ includes loan age. The ramp and plateau were verified by reading the Federal Reserve's [Principal Payments on the Federal Reserve's Securities Holdings](https://www.federalreserve.gov/econres/notes/feds-notes/principal-payments-on-the-federal-reserves-securities-holdings-20170616.html), section **MBS Principal Payments**, paragraph beginning “There are some simple methods”. That section gives the initial annual rate, monthly increment and plateau; footnote 4 also supplies the amortization formula. These are market-model conventions, not regulatory limits.

The rate-dependent CPR is a separate stylized model:

$$\mathrm{CPR}_m=\min(m/M,1)\left[c_0+c_1\max\left(\tanh\frac{c-r_{\mathrm{mort}}}{k},0\right)e^{-\beta u}\right].$$

$u$ is cumulative prepaid principal divided by initial balance, excluding scheduled amortization. $M$ is the seasoning horizon. The mortgage proxy is $r_{\mathrm{mort}}=\max(\mathrm{spread}+\mathrm{loading}\cdot r_m,0)$ at the month-end simulated short rate. These parameters are explicitly stylized and uncalibrated; no empirical validity is asserted. The proxy is not a modeled primary mortgage quote or par mortgage yield. Month-end CPR determines that month's principal, an explicit discrete-time simplification.

Weighted average life is $\mathrm{WAL}=\sum_m t_m\mathrm{Principal}_m/B_0$, in years. Simulation reports the unweighted mean of path WAL, not a discounted duration or price-weighted life. Static PSA cash flows are rate independent, so discounting them with the input curve removes optionality.

## Exact simulation and uncertainty

Over one month $h$, jointly sample the centered endpoint innovation and the integrated innovation with covariance:

$$\begin{pmatrix}v(h)&q(h)\\q(h)&V_I(h)\end{pmatrix}.$$

For independent standard normals $Z_1,Z_2$:

$$x_{t+h}=e^{-ah}x_t+\sqrt{v(h)}Z_1,$$

$$\int_t^{t+h}x_u du=B(t,t+h)x_t+
\frac{q(h)}{\sqrt{v(h)}}Z_1+
\sqrt{V_I(h)-q(h)^2/v(h)}Z_2.$$

Add the exact deterministic shift integral. This jointly simulates Gaussian short rates and path discounts without Euler or trapezoidal discount bias. Prepayment behavior still uses a monthly grid. Zero volatility uses the deterministic limit.

Each random path has a partner using negated innovations. The price standard error is the sample standard deviation of independent pair-average present values divided by the square root of the pair count. Treating each antithetic member as an independent sample would misstate uncertainty. WAL uses the same pairing. OAS standard error is the price error divided by the absolute spread derivative, a local delta-method approximation conditional on model and prepayment assumptions.

The convergence table uses a fixed seed for each path count but paths are not nested across counts. The synthetic MBS market target is generated by the largest displayed run at the reference OAS. Recovery by that same run is a construction identity; smaller runs illustrate sampling differences, not out-of-sample market accuracy. Standard errors exclude calibration, model, prepayment and mortgage-proxy uncertainty. Curve-bump risk uses common random numbers but no risk confidence interval is reported.

## Validation scope

Validation uses closed forms, accounting identities, numerical integration and Gaussian moment identities. There is no QuantLib or external engine comparison. Tests check curve fit, swap repricing, option parity and bounds, calibration recovery, transition moments, state-price conservation, exercise identities, OAS reconciliation, duration benchmarks, amortization conservation, exact Gaussian moments, discounted-bond martingales and Monte Carlo European option agreement. Statistical tests use a predetermined multiple of the estimated standard error; passing does not prove model adequacy.
