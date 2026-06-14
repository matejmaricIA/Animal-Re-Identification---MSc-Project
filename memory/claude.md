# Current Best MegaDescriptor Model

As of 2026-06-05, the current best model is the fine-tuned pretrained MegaDescriptor run:

```text
/lustre/scratch/mmaric2/animal-reid-md/models/megadescriptor_scratch/run06_finetune_pretrained/ckpt_best.pt
```

Evaluation on the WildlifeReID-10k open/closed split:

```json
{
  "closed_set": {
    "db_images": 110848,
    "db_classes": 10236,
    "test_known_images": 22081,
    "top1_acc": 0.7878266382863095,
    "top5_acc": 0.879217426746977
  },
  "open_set": {
    "test_new_images": 7559,
    "max_sim_known_mean": 0.7699782848358154,
    "max_sim_unknown_mean": 0.40476512908935547,
    "unknown_detection_auc": 0.9052460993130327
  }
}
```

This beats the previous pretrained MegaDescriptor baseline:

```text
pretrained top1 ~= 0.7795
pretrained top5 ~= 0.8701
pretrained corrected AUC ~= 0.7558
```

Absolute gains:

```text
top1 +0.0083
top5 +0.0091
AUC  +0.1494
```
