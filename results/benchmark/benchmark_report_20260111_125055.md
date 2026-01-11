# Benchmark Report

**Date:** 2026-01-11 12:50:55

## Environment: gridworld

### Performance Comparison

| Agent | Final Return (Mean ± Std) | 95% CI | Goal Rate |
|-------|---------------------------|--------|-----------|
| sac | 286.51 ± 502.85 | [-294.13, 867.15] | 44.0% |
| vanilla | -36.50 ± 19.63 | [-59.17, -13.83] | 34.7% |
| per | -58.58 ± 11.08 | [-71.38, -45.78] | 14.7% |
| ppo | -73.60 ± 3.60 | [-77.76, -69.44] | 5.3% |


## Key Findings

- **gridworld**: Best agent = `sac` (mean return: 286.51)