import logging
import warnings
import time
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
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


# Training Loop
def train_model(model, train_dataloader, dev_dataloader, epochs, learning_rate):
    optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss, total_train_acc = 0, 0
        start = time.time()
        print(f"Epoch {epoch + 1}/{epochs} - Training Phase")

        for step, batch in enumerate(train_dataloader):
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            labels = batch[2].to(device)

            optimizer.zero_grad()
            loss, prediction = model(input_ids, token_type_ids=None, attention_mask=attention_masks,
                                     labels=labels).values()
            acc = accuracy(prediction, labels)
            total_train_loss += loss.item()
            total_train_acc += acc.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        train_acc = total_train_acc / len(train_dataloader)
        train_loss = total_train_loss / len(train_dataloader)
        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)

        print(f"Epoch {epoch + 1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f}")
        print("Training Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

        # Evaluation
        model.eval()
        total_val_loss, total_val_acc = 0, 0
        print(f"Epoch {epoch + 1}/{epochs} - Evaluation Phase")

        with torch.no_grad():
            for batch in dev_dataloader:
                input_ids = batch[0].to(device)
                attention_masks = batch[1].to(device)
                labels = batch[2].to(device)

                loss, prediction = model(input_ids, token_type_ids=None, attention_mask=attention_masks,
                                         labels=labels).values()
                acc = accuracy(prediction, labels)
                total_val_loss += loss.item()
                total_val_acc += acc.item()

        val_acc = total_val_acc / len(dev_dataloader)
        val_loss = total_val_loss / len(dev_dataloader)
        print(f"Epoch {epoch + 1}: val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}")


# Main Function
def main():
    max_seq_length = 256
    batch_size = 32
    learning_rate = 3e-5
    epochs = 4

    print("Preparing data...")
    train_features, dev_features = prepare_data(tokenizer, max_seq_length)
    train_dataloader, dev_dataloader = create_dataloaders(train_features, dev_features, batch_size)

    print("Initializing model...")
    model = initialize_model()

    print("Starting training...")
    train_model(model, train_dataloader, dev_dataloader, epochs, learning_rate)


if __name__ == "__main__":
    main()
