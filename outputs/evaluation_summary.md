# Model Evaluation Summary

## Dataset
- Test Samples: 51,070
- Default Rate: 11.6%

## Model Performance

- ROC-AUC: 0.7582
- Accuracy (Threshold 0.5): 69.9%
- Accuracy (Tuned Threshold 0.634): 81.5%

### Threshold Comparison

| Metric | Threshold 0.5 | Tuned Threshold |
|--------|--------------:|----------------:|
| Precision | 23.0% | 30.8% |
| Recall | 68.1% | 47.9% |
| F1-score | 0.344 | 0.375 |

## Conclusion

The tuned threshold (0.634) was selected because it improved the F1-score while reducing false positives. Although recall decreased, this threshold provides a better balance between precision and recall for this dataset.
