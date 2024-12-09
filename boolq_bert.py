"""
This script implements a complete pipeline for fine-tuning a BERT-based model for binary sequence classification
(e.g., classifying text pairs into two categories such as True/False or Yes/No). The key functionalities include:

1. Data Loading and Preparation:
   - Loads training and validation datasets from CSV files.
   - Encodes question-passage pairs using a BERT tokenizer into input IDs and attention masks.
   - Prepares PyTorch DataLoader objects for efficient batching and shuffling during training.

2. Model Initialization:
   - Initializes a pre-trained BERT model for sequence classification with two output labels.
   - Moves the model to the appropriate computing device (CPU/GPU).

3. Training and Evaluation:
   - Implements a training loop that performs:
     - Forward pass through the model to compute predictions and loss.
     - Backpropagation and gradient clipping.
     - Parameter updates using the AdamW optimizer and a learning rate scheduler.
   - Includes an evaluation loop to calculate validation loss and accuracy.
   - Supports early stopping based on validation loss to prevent overfitting.

4. Model Saving:
   - Saves the best-performing model during training to a file.
   - Includes functionality to save the model's full state for later resumption or reuse.

5. Main Pipeline:
   - Combines data preparation, model initialization, training, and evaluation in an end-to-end workflow.

The script is designed to work with the BoolQ dataset but can be adapted for other binary classification tasks
with minimal changes to the dataset format or pre-processing steps.
"""


import logging
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

# Suppress warnings and unnecessary logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load BERT Tokenizer
def load_tokenizer():
    print('Loading BERT tokenizer...')
    return BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


tokenizer = load_tokenizer()


# Load Dataset
def load_data(tokenizer, questions, passages, max_length):
    """Encode the question/passage pairs into features that can be fed to the model."""
    input_ids = []
    attention_masks = []

    for question, passage in zip(questions, passages):
        encoded_data = tokenizer.encode_plus(question, passage, max_length=max_length, pad_to_max_length=True,
                                             truncation_strategy="longest_first")
        encoded_pair = encoded_data["input_ids"]
        attention_mask = encoded_data["attention_mask"]

        input_ids.append(encoded_pair)
        attention_masks.append(attention_mask)
    return np.array(input_ids), np.array(attention_masks)


def prepare_data(tokenizer, max_seq_length):
    # Load datasets
    train_data_df = pd.read_csv("./dataset/BoolQ/train.csv", header=None, names=('question', 'passage', 'label', 'idx'))
    dev_data_df = pd.read_csv("./dataset/BoolQ/val.csv", header=None, names=('question', 'passage', 'label', 'idx'))

    # Extract training data
    passages_train = train_data_df.passage.values
    questions_train = train_data_df.question.values
    answers_train = train_data_df.label.values.astype(int)

    # Extract validation data
    passages_dev = dev_data_df.passage.values
    questions_dev = dev_data_df.question.values
    answers_dev = dev_data_df.label.values.astype(int)

    # Encoding data
    input_ids_train, attention_masks_train = load_data(tokenizer, questions_train, passages_train, max_seq_length)
    input_ids_dev, attention_masks_dev = load_data(tokenizer, questions_dev, passages_dev, max_seq_length)

    return (input_ids_train, attention_masks_train, answers_train), (input_ids_dev, attention_masks_dev, answers_dev)


def create_dataloaders(train_features, dev_features, batch_size):
    train_features_tensors = [torch.tensor(feature, dtype=torch.long) for feature in train_features]
    dev_features_tensors = [torch.tensor(feature, dtype=torch.long) for feature in dev_features]

    train_dataset = TensorDataset(*train_features_tensors)
    dev_dataset = TensorDataset(*dev_features_tensors)

    train_sampler = RandomSampler(train_dataset)
    dev_sampler = SequentialSampler(dev_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=batch_size)

    return train_dataloader, dev_dataloader


# Initialize Model
def initialize_model():
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=2,  # The number of output labels--2 for binary classification.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    model.to(device)
    return model


# Accuracy Calculation
def accuracy(y_pred, y_test):
    acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test).sum().float() / float(y_test.size(0))
    return acc


def train_model(model, train_dataloader, dev_dataloader, epochs, learning_rate):
    optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

    best_val_loss = float('inf')
    no_improve_epochs = 0
    early_stopping_threshold = 3  # Number of epochs to continue without improvement in validation loss

    for epoch in range(epochs):
        model.train()
        total_train_loss, total_train_acc = 0, 0
        for batch in tqdm(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_masks, labels = batch

            model.zero_grad()
            outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks, labels=labels)
            loss = outputs[0]
            logits = outputs[1]

            acc = accuracy(logits, labels)
            total_train_loss += loss.item()
            total_train_acc += acc.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_acc = total_train_acc / len(train_dataloader)

        # Evaluation phase
        model.eval()
        total_val_loss, total_val_acc = 0, 0
        for batch in dev_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_masks, labels = batch

            with torch.no_grad():
                outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks, labels=labels)
                loss = outputs[0]
                logits = outputs[1]

            acc = accuracy(logits, labels)
            total_val_loss += loss.item()
            total_val_acc += acc.item()

        avg_val_loss = total_val_loss / len(dev_dataloader)
        avg_val_acc = total_val_acc / len(dev_dataloader)

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), 'model_best.pt')  # Save the best model
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_threshold:
                print("Early stopping triggered.")
                break


def save_model_full(model, optimizer, epoch, file_path):
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    torch.save(state, file_path)


# Main Function
def main():
    max_seq_length = 256
    batch_size = 16
    learning_rate = 2e-5
    epochs = 5

    print("Preparing data...")
    train_features, dev_features = prepare_data(tokenizer, max_seq_length)
    train_dataloader, dev_dataloader = create_dataloaders(train_features, dev_features, batch_size)

    print("Initializing model...")
    model = initialize_model()

    print("Starting training...")
    train_model(model, train_dataloader, dev_dataloader, epochs, learning_rate)


if __name__ == "__main__":
    main()
