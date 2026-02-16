# Methods Guide

## IID Methods

### Percentile
Standard quantile-based CI. Simple but can have poor coverage for skewed distributions.

### Basic (Reverse Percentile)
$$CI = [2\hat{\theta} - q_{1-\alpha/2},\; 2\hat{\theta} - q_{\alpha/2}]$$

### BCa (Bias-Corrected and Accelerated)
The recommended default. Corrects for bias and skewness using jackknife acceleration:

$$\hat{a} = \frac{\sum_{i=1}^{n}(\hat{\theta}_{(\cdot)} - \hat{\theta}_{(i)})^3}{6\left[\sum_{i=1}^{n}(\hat{\theta}_{(\cdot)} - \hat{\theta}_{(i)})^2\right]^{3/2}}$$

### Studentized (Bootstrap-t)
Uses pivotal quantity $t^* = (\hat{\theta}^* - \hat{\theta}) / \hat{se}^*$ with nested bootstrap for SE.

### Poisson Bootstrap
Weighted resampling with $W \sim \text{Poisson}(1)$. Ideal for streaming/online algorithms.

### Bernoulli Bootstrap
Binary weights $W \sim \text{Bernoulli}(p)$. Useful for specific ML applications.

### Subsampling (m-out-of-n)
Sampling without replacement, size $m < n$. Required for non-regular statistics (max, min).

---

## Time Series Methods

### Moving Block Bootstrap (MBB)
Overlapping fixed-length blocks. Set `block_length` based on autocorrelation structure.

### Circular Block Bootstrap (CBB)
Wraps data circularly to eliminate edge effects. Same block logic as MBB.

### Stationary Bootstrap (Politis & Romano)
Random block lengths $L \sim \text{Geometric}(1/\bar{L})$ where $\bar{L}$ = `mean_block`.

### Tapered Block Bootstrap
Applies a tapering window (Tukey, Hanning, etc.) to each block. For spectral density estimation.

### AR-Sieve Bootstrap
Fits AR(p) model → extracts residuals → resamples residuals → reconstructs series.

### Wild Bootstrap
$$y_t^* = \hat{y}_t + \hat{\varepsilon}_t \cdot v_t$$
where $v_t$ is Rademacher (±1) or Mammen two-point. Handles heteroskedasticity.

---

## Hierarchical Methods

### Cluster Bootstrap
Resamples entire clusters (groups), preserving within-group correlation structure.

### Stratified Bootstrap
Resamples within each stratum independently, preserving class proportions.
