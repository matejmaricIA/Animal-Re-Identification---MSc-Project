# Counting command cookbook (calibrated + power setup)

Use this file as a quick copy/paste reference for count runs with the reduced feature set:
- calibrated or power proposal mode
- global embeddings + Fisher vectors
- optional strict confirmation voting for "same" labels

## Quick setup

```bash
DS=seastarreid2023
EMB=megadescriptor-l-384
SEED=0
```

## A) Standard calibrated run (global + fisher)

```bash
python main.py --count --ds $DS \
  --use_global_embedding --embedding_model $EMB \
  --use_fisher --method ensamble \
  --remove_background \
  --count_proposal_mode calibrated \
  --num_vertices 100 --num_neighbors 800 \
  --seed $SEED --save_count
```

## B) Raw-score run (skip calibration)

```bash
python main.py --count --ds $DS \
  --use_global_embedding --embedding_model $EMB \
  --use_fisher --method ensamble \
  --remove_background \
  --count_proposal_mode calibrated \
  --count_skip_calibration \
  --num_vertices 100 --num_neighbors 800 \
  --seed $SEED --save_count
```

## C) Power-mode run

```bash
python main.py --count --ds $DS \
  --use_global_embedding --embedding_model $EMB \
  --use_fisher --method ensamble \
  --remove_background \
  --count_proposal_mode power \
  --num_vertices 100 --num_neighbors 800 \
  --seed $SEED --save_count
```

## D) Force recalibration

```bash
python main.py --count --ds $DS \
  --use_global_embedding --embedding_model $EMB \
  --use_fisher --method ensamble \
  --remove_background \
  --count_proposal_mode calibrated \
  --count_force_recalibrate \
  --count_cal_pairs 2500 \
  --count_cal_shortlist 150 \
  --count_cal_negs_per_query 400 \
  --num_vertices 100 --num_neighbors 800 \
  --seed $SEED --save_count
```

## E) Label-error robustness (strict same-vote confirmation)

```bash
python main.py --count --ds $DS \
  --use_global_embedding --embedding_model $EMB \
  --use_fisher --method ensamble \
  --remove_background \
  --count_proposal_mode calibrated \
  --count_confirm_same_votes 3 \
  --num_vertices 100 --num_neighbors 800 \
  --seed $SEED --save_count
```

## F) Error-rate sweep example (single dataset)

```bash
bash test-scripts/run_elpephants_error_rate_analysis.sh
```

## Argument cheatsheet (plain English)

- `--count`: run population counting (HITL-NIS).
- `--ds`: dataset name.
- `--use_global_embedding`: include global embedding signal.
- `--embedding_model`: global model (`resnet50` or `megadescriptor-l-384`).
- `--use_fisher`: include Fisher-vector signal.
- `--method`: Fisher feature extractor (`disk`, `superpoint`, `aliked`, or `ensamble`).
- `--count_proposal_mode`: `calibrated` or `power`.
- `--count_force_recalibrate`: retrain count calibrators instead of loading cache.
- `--count_skip_calibration`: bypass calibrators and use raw global/Fisher similarity mapping.
- `--count_cal_pairs`: number of pairs used for calibration.
- `--count_cal_shortlist`: shortlist size used when sampling hard negatives for calibration.
- `--count_cal_negs_per_query`: negatives per query in calibration.
- `--count_confirm_same_votes K`: require K consecutive "same" votes to accept a pair as same. `K=1` disables.
- `--num_vertices`: number of outer-loop sampled vertices.
- `--num_neighbors`: sampled neighbors per vertex.
- `--label_error_rate`: simulated oracle flip probability for robustness tests.
- `--seed`: random seed.
- `--save_count`: append result row to counting results XLSX.
