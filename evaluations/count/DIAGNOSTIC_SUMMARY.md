# Quick Diagnostic Summary

## The Core Problem

```
Ground Truth: 48 individuals
Estimated:    126 individuals (+163%)
```

**Translation**: Your estimator thinks there are **2.6× more unique individuals** than there actually are.

---

## Why This Happens in NIS

### The Math
```
N̂ = Σ (1/Q[u]) × (1/(1 + d̂[u]))
```

Where:
- `Q[u]` = probability of sampling vertex u
- `d̂[u]` = estimated degree (# of edges to same individual)

### What Should Happen (Ground Truth)
- 48 individuals, 1086 images → ~22.6 images per individual
- Each vertex has ~21.6 edges to same-ID images (degree)
- NIS samples pairs, oracle says "same" 21.6/1086 = 2% of the time
- Formula: `1/(1+21.6) ≈ 0.044` → multiply by 1086 → ~48 ✅

### What's Actually Happening
- Estimator gets 126 → implies avg degree of ~7.6
- This means oracle is saying "same" much less often than expected
- **Why?** Proposal is sampling the **wrong pairs**

---

## The Calibration Problem Visualized

### Your Calibration Set
```
500 pairs: 128 positives (25.6%) + 372 negatives (74.4%)
```

### Reality
```
Total possible pairs: 1086 × 1085 / 2 = 589,155
True matches: ~48 × 22 = 1,056 
Positive rate: 1,056 / 589,155 = 0.18% ❌
```

**Mismatch**: Calibration is **140× richer** in positives than the full graph!

### What This Does to Your Proposal

1. **Isotonic calibrator sees**: "When global sim = 0.7, there's a 40% chance of match" (based on calibration data)
2. **Reality**: "When global sim = 0.7, there's a 5% chance of match" (actual population)
3. **Proposal assigns**: `q(u,v) = 0.4` for these pairs (too high!)
4. **Importance weight**: `1/q(u,v) = 2.5` (too low!)
5. **Net effect**: True matches don't get enough weight, non-matches get too much

### The Vicious Cycle

```
High Cal Pos Rate
    ↓
Calibrator Maps Scores Too High
    ↓
Proposal q(u,v) Too Optimistic
    ↓
Samples Many Plausible Non-Matches
    ↓
Oracle Says "Different" Often
    ↓
Degree Estimates Too Low
    ↓
Population Estimate Too High ❌
```

---

## The Fix: Rebalance Calibration

### Current Strategy (Problematic)
```python
for query in calibration_queries:
    positives = ALL images with same identity  # ← Problem!
    negatives = 100 random from shortlist
    pairs += (query, positives) + (query, negatives)
```

**Result**: Massively overweights positives.

### Better Strategy
```python
for query in calibration_queries:
    positives = min(2, num_same_identity)  # ← Cap at 2-3
    negatives = 200 random from shortlist  # ← More negatives
    pairs += (query, positives) + (query, negatives)
```

**Result**: Positive rate ~1-5% (closer to reality).

---

## Immediate Actions

### 1️⃣ Quick Win: Adjust Hyperparameters
```bash
python main.py --count --ds chicks4freeid \
  --count_cal_pairs 1000 \              # More data
  --count_shortlist_B 150 \             # Smaller shortlist (more selective)
  --count_mix_alpha 0.7 \               # Less trust in shortlist
  --num_vertices 250 \                  # More samples
  --save_count
```

### 2️⃣ Code Fix: Rebalance Calibration
Edit `calibration.py::build_calibration_pairs_stratified()`:

```python
# Around line 165-169
positives = [i for i in all_ids if train_labels[i] == q_identity and i != q_id]

# ADD THIS CAP:
positives = random.sample(positives, min(3, len(positives)))  # ← NEW

for p_id in positives:
    query_ids.append(q_id)
    db_ids.append(p_id)
    pair_labels.append(1)
```

### 3️⃣ Validation: Check Other Datasets
Test on 3-5 other WildlifeReID10k datasets to see if the problem is:
- **Systematic** (all datasets overestimate) → Calibration issue
- **Dataset-specific** (chicks are hard) → Feature quality issue

---

## Expected Outcomes After Fixes

### If Calibration Was the Problem
- Estimate: 40-55 (within ~15% of ground truth)
- CI: [35, 60] (ground truth inside)
- Stderr: 5-10 (tighter)

### If Sample Size Was the Problem
- Estimate: Still ~120, but CI wider [60, 180]
- Ground truth now in CI, but estimate still biased

### If Features Are Bad
- No improvement even with fixes
- Need to revisit feature quality / embedding discriminability

---

## Red Flags to Monitor

❌ **Cal Pos Rate > 15%** → Calibration too imbalanced  
❌ **Redundancy > 10%** → Sampling from too narrow a region  
❌ **Cache Hit Rate < 5%** → Shortlist not overlapping (bad vertex ordering)  
❌ **CI Width > 100% of estimate** → Undersampled or bad proposal  
❌ **GT outside CI** → Bias or severe underestimation of uncertainty  

---

## Theoretical Note

**Is the estimator still unbiased?**

**Yes**, in principle:
- Full support: ✅ (q_eps + base proposal)
- Importance weighting: ✅ (correct formula)
- Oracle is ground truth: ✅ (no noise)

**But**:
- **Practical bias** can emerge from poor proposal (even if theoretically unbiased)
- **Variance can be enormous** if proposal doesn't match target
- You're likely in a **high-variance regime** that requires 10× more samples to converge

**Analogy**: It's like estimating the mean of a distribution by importance sampling from a very different distribution. Unbiased? Yes. Practical? No.

---

**TL;DR**: Your calibration set has 140× more positives than reality, making the proposal too optimistic, which makes the estimator think there are more individuals than there are. Fix: rebalance calibration to ~2-5% positive rate.
