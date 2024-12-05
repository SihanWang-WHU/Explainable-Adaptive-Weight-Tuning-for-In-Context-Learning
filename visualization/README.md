For writing the results section, I used three pretrained models for fine-tuning and result visualization. The models are:

- checkpoint_path = "Intel/bert-base-uncased-squadv1.1-sparse-80-1x4-block-pruneofa"
- checkpoint_path = "armageddon/bert-base-uncased-squad2-covid-qa-deepset"
- checkpoint_path = "dccuchile/bert-base-spanish-wwm-uncased-finetuned-qa-mlqa"

To correspond to these three models, I assigned three postfixes to name the results:

- "squad"
- "covid"
- "spanish"

The sequence of models, squad → covid → spanish, reflects an increasing distance from the original fine-tuning task, which can be observed through weight visualizations.

**From the evaluation results, it is clear that the model fine-tuned on squad performs best across all evaluation metrics, followed by the covid model, and finally, the spanish model.**
