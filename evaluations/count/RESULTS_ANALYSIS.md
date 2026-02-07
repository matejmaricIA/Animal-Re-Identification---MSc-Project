# Population Counting Results Analysis

## Executive Summary

**Dataset**: chicks4freeid (WildlifeReID10k)  
**Ground Truth**: 48 individuals  
**Estimated**: 126.21 ± 20.17 individuals (95% CI: [86.67, 165.75])  
**Error**: +78.21 individuals (+162.9% overestimation)  

**⚠️ CRITICAL ISSUE**: The estimator is **severely overestimating** the population by more than 2.6×.

---

## Detailed Analysis

### 1. Accuracy Assessment: ❌ POOR

**Problem**: The ground truth (48) falls **well outside** the 95% confidence interval [86.67, 165.75].

**What this means**:
- Either the estimator is **biased** (systematic error)
- Or you encountered an **extremely unlucky** sample (< 5% probability)
- The confidence intervals may be **underestimated** (see stderr concerns below)

**Likely causes**:
1. **Proposal distribution too diffuse**: High `mix_alpha=0.9` with poor shortlist quality → sampling wrong pairs
2. **Calibration mismatch**: 25.6% positive rate in calibration vs. true ~4.4% (48 positives out of ~1086 images)
3. **Local evidence misleading**: Inliers may be giving false positives on similar-looking chicks

---

### 2. Precision Assessment: ⚠️ MODERATE

**CI Width**: 79.08 (62.7% of estimate) is **very wide** but given the overestimation, this is misleading.

**Coefficient of Variation**: 15.98% would be acceptable **if the estimate were accurate**, but it's not.

**Interpretation**: The estimator thinks it's uncertain (wide CI), but it's systematically wrong (biased high).

---

### 3. Calibration Quality: ⚠️ CONCERNING

```
Cal Pairs:     500 (128 pos / 372 neg)
Positive Rate: 25.6%
True Rate:     ~4.4% (48 identities / 1086 images ≈ 590,685 pairs)
```

**Problem**: Your calibration set has a **6× higher positive rate** than reality.

**Why this happens**:
- You sample calibration queries and then find **all positives** for each query
- This creates an **imbalanced calibration set** that overweights matches
- The shortlist sampling (300 top candidates) helps but doesn't fully fix this

**Impact on proposal**:
- Calibrators learn to map scores → high probabilities
- Proposal `q(u,v)` becomes too "optimistic" about matches
- NIS samples pairs that look similar but aren't actual matches
- Importance weights `1/q(u,v)` are too small → degree estimates too high → population estimate too high

---

### 4. Efficiency Metrics: ✅ GOOD

```
Oracle Calls:     3,000
Unique Pairs:     2,883 (redundancy: 3.9%)
Local Attempts:   39,580
Local Cache Hits: 5,420 (12% hit rate)
Runtime:          50.79 minutes
```

**Positives**:
- ✅ Low redundancy (3.9%) means good diversity in sampled pairs
- ✅ Caching is working (12% hit rate reduces ~5K GV computations)
- ✅ Runtime is reasonable (~1 minute per vertex)

**Observations**:
- 39,580 local attempts for 300-sized shortlist × 150 vertices = ~264 per vertex
  - This suggests shortlist overlap across vertices (good for caching)
- Cache hit rate could be higher with better vertex ordering or larger alpha

---

### 5. Configuration Assessment

**Proposal Mode: calibrated** ✅ Correct choice  
**Signals: Global + Fisher + Local** ✅ Comprehensive  
**Embedding Model: MegaDescriptor-L-384** ✅ State-of-the-art  
**Fisher Method: ensemble** ✅ Multi-scale  
**GV Matcher: LightGlue** ✅ Best available  

**BUT**:
- `Shortlist B = 300` may be too large for 1086 images (27% of dataset)
- `Mix α = 0.9` places too much trust in shortlist quality
- `Local µ = 0.5` might be too permissive for inliers

---

## Root Cause Diagnosis

### Most Likely Culprit: **Calibration Positive Rate Mismatch**

The 25.6% calibration positive rate vs. ~4.4% true rate means:

1. **Calibrators are too optimistic**: They map scores to probabilities that are 6× too high
2. **Proposal samples "plausible but wrong" pairs**: High-similarity non-matches get high `q(u,v)`
3. **Importance weights are too small**: `1/q(u,v)` underweights these pairs
4. **Degree estimates are inflated**: Each vertex appears more connected than it is
5. **Population estimate is too high**: NIS formula `1/(1+degree)` shrinks too much

### Mathematical Intuition

In NIS, the estimate is:
```
N̂ = (1/n_vertices) × Σ (1/Q[u]) × (1/(1 + d̂[u]))
```

Where `d̂[u]` is the estimated degree (number of same-individual edges).

If the proposal makes non-matches look like matches:
- `d̂[u]` is artificially high (many false positives)
- `1/(1 + d̂[u])` becomes small
- But this should make `N̂` **smaller**, not larger!

**Wait...** 🤔 This suggests the problem might be different:

### Alternative Hypothesis: **Proposal is TOO FLAT**

If calibration is **too conservative** (opposite of what I said):
- Many pairs get similar low probabilities
- Proposal `q(u,v)` is nearly uniform
- True matches get **large** importance weights `1/q(u,v)`
- But false matches also get large weights
- Net effect: overestimation

Let me recalculate: With 48 true individuals, avg degree should be ~(1086/48 - 1) ≈ 21.6 edges per vertex.

If estimator gets ~126 individuals, it thinks avg degree is ~(1086/126 - 1) ≈ 7.6 edges per vertex.

So it's **underestimating degrees**, which means it thinks there are **more disconnected components** (more individuals).

---

## Revised Root Cause

**The proposal is not discriminative enough**:
- It assigns similar probabilities to matches and non-matches
- True matches don't get sufficiently high `q(u,v)` to dominate the sample
- The estimator samples many "borderline" pairs that turn out to be non-matches
- These contribute to low degree estimates → overestimated population

**Evidence**:
- 25.6% positive rate in calibration (should be ~4.4%)
- This suggests shortlist contains many hard negatives that calibrator can't distinguish
- Isotonic regression maps these ambiguous scores to mid-range probabilities
- Proposal becomes "mushy" instead of peaked at true matches

---

## Recommended Fixes (Priority Order)

### 🔴 **HIGH PRIORITY**

1. **Rebalance Calibration Set**
   - Target positive rate closer to true rate (4-10%)
   - Sample **fewer** positives per query (e.g., max 2-3 instead of all)
   - Sample **more** hard negatives from shortlist (current: 100, increase to 200-300)
   - Target: 1000-2000 calibration pairs total

2. **Reduce Mix Alpha**
   - Current: 0.9 (90% shortlist weight)
   - Try: 0.7-0.8 (70-80% shortlist weight)
   - Gives more "support mass" to base proposal for robustness

3. **Increase Number of Vertices**
   - Current: 150 vertices × 20 neighbors = 3,000 samples
   - Try: 200-300 vertices (4,000-6,000 samples)
   - Better coverage of the graph → tighter CI

### 🟡 **MEDIUM PRIORITY**

4. **Reduce Shortlist Size**
   - Current: 300 (27% of images)
   - Try: 100-150 (10-15% of images)
   - Forces shortlist to be more selective → better calibration

5. **Tighten Local µ Threshold**
   - Current: 0.5 (fairly permissive)
   - Try: 0.7-0.8 (stricter match confidence)
   - Reduces false positive inliers

6. **Use Power Rule Instead**
   - Switch to `--count_proposal_mode power` as ablation
   - Bypasses calibration issues
   - May give cleaner results if calibration is the problem

### 🟢 **LOW PRIORITY / DIAGNOSTIC**

7. **Inspect Calibration Curves**
   - Plot calibrated P(match | score) vs. score for each signal
   - Check if curves are monotonic and well-separated
   - Look for plateau regions (poor discrimination)

8. **Bootstrap Stderr**
   - Current stderr assumes i.i.d. (may underestimate)
   - Implement block bootstrap over vertices
   - Get more realistic CI widths

9. **Adaptive Q(u)**
   - Switch from uniform Q(u) to degree-based sampling
   - Focus on high-degree nodes (more informative)

---

## Validation Experiments

### Experiment 1: Ground Truth Sanity Check
```bash
python main.py --count --ds chicks4freeid \
  --count_proposal_mode calibrated \
  --count_cal_pairs 1000 \
  --count_shortlist_B 150 \
  --count_mix_alpha 0.7 \
  --num_vertices 250 \
  --num_neighbors 20 \
  --save_count
```

**Expected**: Estimate closer to 48, CI narrower, ground truth in CI.

### Experiment 2: Power Rule Baseline
```bash
python main.py --count --ds chicks4freeid \
  --count_proposal_mode power \
  --count_shortlist_B 150 \
  --count_mix_alpha 0.7 \
  --num_vertices 250 \
  --num_neighbors 20 \
  --save_count
```

**Expected**: If this is more accurate, confirms calibration is the issue.

### Experiment 3: More Datasets
Run on other WildlifeReID10k datasets with known ground truth:
- ATRW (Tigers)
- SeaStarReID
- BelugaID

**Expected**: Check if overestimation is consistent or dataset-specific.

---

## Theoretical Soundness Re-Assessment

### What's Still Sound ✅
- Unbiasedness in principle (full support via q_eps + base distribution)
- Late fusion in proposal only
- Importance weighting formula
- No GV gating (every sample queries oracle)

### What Might Break Soundness ⚠️
- **Calibration-proposal feedback loop**: If calibrators are trained on a biased sample (high positive rate), the resulting proposal might systematically favor certain regions of the graph
- **Shortlist selection bias**: Top-B by base similarity might exclude true positives with low global/Fisher similarity
- **Local evidence ambiguity**: Chicks might have similar patterns → many false positive inliers → local signal misleads

### Is This Still Unbiased?

**Technically yes**, but:
- Unbiasedness holds if `q(u,v) > 0` for all pairs ✅ (you have this via q_eps)
- But **variance can be infinite** if proposal is very poorly matched to truth
- You might be in a "high variance regime" where 150 samples isn't enough

---

## Final Recommendation

**Immediate Action**: Run Experiments 1 and 2 above to diagnose whether:
- **Calibration is the issue** (Exp 2 > Exp 1) → Fix calibration positive rate
- **Sample size is the issue** (Exp 1 ≈ current but with wider CI) → Increase vertices
- **Dataset is fundamentally hard** (Both experiments fail) → Check feature quality

**Long-term**: Once you get accurate estimates on chicks4freeid, validate across 5-10 datasets to ensure the fix generalizes.

---

## Questions for Further Investigation

1. What does the degree distribution look like? (Are there outlier vertices with very high degree?)
2. What's the empirical match rate in the 3,000 oracle calls? (Should be ~4-5%)
3. Are there "confuser" identities (e.g., pairs of chicks that look very similar)?
4. Does the global embedding cluster correctly? (Try t-SNE plot of embeddings colored by identity)
5. What's the precision/recall of the base proposal before local evidence? (Top-300 shortlist accuracy)

---

**Generated**: 2026-02-05  
**Dataset**: chicks4freeid (1086 images, 48 identities)  
**Configuration**: Calibrated proposal, MegaDescriptor + Ensemble Fisher + LightGlue  
