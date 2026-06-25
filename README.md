# F1 Fantasy Team Optimization Using Contextual Linear Bandits

Automatic racing team selection for Fantasy Formula 1 using contextual linear bandits (LinUCB). The algorithm learns which drivers and constructors to pick each race by balancing exploration of uncertain options with exploitation of proven performers.

## Overview

Fantasy Formula 1 requires selecting 5 drivers and 2 constructors each week to maximize points across a 24-race season, subject to a $100 million budget. This project frames the problem as a sequential decision-making task where an algorithm learns from race outcomes and improves over time.

## Results

**Enhanced Bandit vs Baselines (48 test races, 2024-2025):**
- Enhanced Bandit: **8,933 points**
- Human baseline: 8,485 points (+5.3%)
- Standard Bandit: 8,585 points (+4.1%)
- Greedy baseline: 6,272 points (-29.8%)

**Key Finding:** Enhanced Bandit (full feedback) has tunable hyperparameters, while Standard Bandit (limited feedback) shows high variance across runs, making optimization difficult despite potentially beating Enhanced.

## Approach

### Algorithm
**LinUCB (Linear Upper Confidence Bound)** estimates expected points as a linear combination of features:
```
expected_points = θ^T × features
ucb_score = expected_points + α × √(features^T × A^(-1) × features)
```

Maintains confidence bounds to balance exploration (trying uncertain drivers) vs exploitation (picking proven performers).

### Learning Strategies
- **Standard Bandit:** Updates θ only for selected drivers (7 observations/race)
- **Enhanced Bandit:** Updates θ for all drivers (30 observations/race)

### Features (46 total)

**Driver Features (28):**
- Performance history: Recent form, overall average, track-specific, trend, consistency
- Cost dynamics: Current cost, cost changes
- Driver attributes: Experience, age, championships, podiums
- Environmental: Track characteristics (length, turns, DRS zones, type)
- Signals: Sprint weekend indicator
- Driver of the Day: Win rate, streak, total wins

**Constructor Features (18):**
- Similar structure with team-level aggregates

Features use recency weighting (exponential decay, weighted averages) and are normalized to [-1.5, 1.5] ranges.

### Team Selection
Exhaustive search over all valid team combinations (C(20,5) × C(10,2)) with budget pruning. Applies 2× multiplier to highest-UCB driver per F1 Fantasy rules.

## Key Findings

1. **Performance history dominates** - Recent form (4.27) and overall average (2.44) account for top predictors
2. **Sprint weekends matter** - Sprint race signal has weight 2.77, second most important
3. **Track-specific performance surprisingly weak** - Weight only 0.09 despite intuition
4. **Hyperparameter stability differs** - Enhanced shows consistent tuning effects; Standard varies wildly with seed
5. **Limited feedback creates variance** - Standard's randomness occasionally outperforms Enhanced (best run: 8,969 vs 8,933) but is unreliable

## Optimal Hyperparameters

| Strategy | Alpha | Lambda |
|----------|-------|--------|
| Enhanced | 0.7 | 1.0 |
| Standard | 0.7 | 3.5 |

Both require identical exploration (α=0.7) but differ in regularization, suggesting limited feedback needs stronger regularization despite less data.

## Usage

```python
# Initialize bandit for a driver
bandit = ContextualBandit(n_features=28, alpha=0.7, lambda=1.0)

# Get predictions
expected_points, ucb_score = bandit.predict(driver, features)

# Update after observing results
bandit.update(driver.name, features, actual_points)

# Find best team
best_team = find_best_team(drivers, constructors, bandit, budget=100_000_000)
```

## Files

- `fantasyf1__2_.ipynb` - Main notebook with algorithm implementation and evaluation
- `f1_fantasy_report_for_word.md` - Detailed technical report (IEEE format)
- `feature_importance_analysis.png` - Visualization of learned feature weights

## Requirements

- Python 3.7+
- NumPy
- Pandas
- scikit-learn
- Matplotlib

## Future Work

**Algorithmic:**
- Add randomization to Enhanced to escape local optima
- Implement Thompson Sampling for alternative exploration
- Develop Dueling Bandits approach for direct driver comparisons

**Feature Engineering:**
- Remove low-weight features (17/46 have |θ| < 0.5), reduce from 46 to 29
- Add team transfer indicators
- Include tire strategy information

**Deployment:**
- Evaluate on 2026 season in real-time
- Compare against other human players (current: single baseline)

## Contact & Citation

For questions or collaboration, reach out through GitHub.

If using this work, cite as:
```
[Your Name]. F1 Fantasy Team Optimization Using Contextual Linear Bandits. 2026.
```
