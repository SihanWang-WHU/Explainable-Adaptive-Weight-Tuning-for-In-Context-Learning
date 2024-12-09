# Explainable Adaptive Weight Tuning for Language Models

This repository implements Explainable Adaptive Weight Tuning to improve the fine-tuning and interpretability of Language Models like BERT. By combining fine-tuning techniques with explainable AI (XAI) methods such as LIME and gradient weighting, this project enhances model performance and transparency. The goal is to provide a more effective, interpretable approach to LLM fine-tuning for domain-specific tasks, fostering trust and accountability in AI systems.

## Methodology

The methodology behind this project combines model fine-tuning, gradient weighting and explainability to achieve higher accuracy and interpretability in LMs. The key steps of our approach are:

### 1. Fine-Tuning BERT
We fine-tuned a pre-trained BERT model on benchmark datasets, including SuperGLUE's BoolQ dataset and SQuAD for question-answering tasks. This fine-tuning process involves training the model on task-specific data to adapt it to the nuances of the task while improving performance on unseen data.

![bert_structure](https://github.com/user-attachments/assets/a34adbb7-fe66-4a89-a283-b85e4ce14ad8)


### 2. Explainable AI with LIME
We integrated the LIME framework to provide local, interpretable explanations for the predictions made by the fine-tuned BERT model. LIME helps us identify which tokens are most influential in the model’s decision-making process, improving model transparency and trustworthiness.

### 3. Gradient Weighting
To further enhance the fine-tuning process, we used ExAWT to apply adaptive gradient weighting. This method prioritizes layers and components of the model based on their contribution to contextual understanding, improving the efficiency and effectiveness of fine-tuning on specific tasks.

## Datasets

We used the following publicly available datasets for evaluating the effectiveness of our approach:

### 1. SuperGLUE
SuperGLUE is a widely used benchmark for evaluating advanced natural language understanding (NLU) models. It provides various challenging tasks that require nuanced reasoning, including BoolQ, which was selected for this study. The BoolQ dataset focuses on binary question-answering tasks, providing 9,427 training examples and 3,270 validation examples.

### 2. SQuAD
The Stanford Question Answering Dataset (SQuAD) is a large-scale dataset used for training and evaluating question-answering systems. We used both versions of the dataset (SQuAD v1.0 and SQuAD v2.0) to assess the robustness of our fine-tuned models in handling answerable and unanswerable questions.

By leveraging these datasets, we were able to rigorously test the fine-tuning process and evaluate the impact of ExAWT and LIME on model accuracy and interpretability.

## Results

The results of our experiments show significant improvements in both model performance and interpretability after fine-tuning using the proposed methods.

### Performance Metrics
We used the following evaluation metrics to assess the models:

- **Training Loss**: Measures the convergence speed during fine-tuning.
- **Exact Match (EM)**: Represents the percentage of predictions that exactly match the ground truth.
- **F1 Score**: Balances precision and recall, providing a more nuanced evaluation for tasks with imbalanced datasets.

### Key Findings

- The **SQuAD QA** model achieved the highest Exact Match (EM) score of 70.05%, demonstrating its effectiveness in general question-answering tasks.
- The **COVID QA** model, fine-tuned on domain-specific data, performed closely to the SQuAD model with an EM score of 69.89%.
- The **Spanish QA** model faced slight challenges in language transfer, with a slightly lower EM score of 69.68%.

![lime_explaination](https://github.com/user-attachments/assets/9b8cea56-be12-4b9a-b27c-3a64bcd0279f)


### Visualizing LIME Interpretations

Using the LIME framework, we observed that fine-tuning BERT resulted in more meaningful explanations. For instance, in the BoolQ task, the fine-tuned BERT model focused on relevant tokens like “Paris” and “capital,” leading to correct predictions with high confidence (99%). In contrast, the base model incorrectly focused on irrelevant tokens, leading to mispredictions.

![eval_all](https://github.com/user-attachments/assets/63598fdf-0904-4626-96db-9b9633ed93d0)


## Conclusion

In conclusion, this project demonstrates that Explainable Adaptive Weight Tuning (ExAWT) improves both the performance and interpretability of Large Language Models like BERT. The integration of fine-tuning with explainable AI methods, such as LIME, helps models focus on task-relevant features and provides clear justifications for their predictions. Our findings suggest that these methods offer a robust approach to developing more transparent, reliable, and ethically aligned AI systems.

Future work can explore further domain-specific applications, refine the gradient weighting techniques, and test the scalability of these methods with more complex models.


