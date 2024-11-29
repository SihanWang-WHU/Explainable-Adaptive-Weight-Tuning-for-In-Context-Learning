import json

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_scheduler,
    glue_processors
)
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import BertTokenizer
from tqdm import tqdm


def load_finetune_dataset(task_name='sst2', max_length=128):
    """
    Automatically download and prepare the SST-2 dataset using the Hugging Face datasets library.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Download SST-2 dataset
    dataset = load_dataset('glue', task_name)

    # Split into train and validation sets
    train_texts = dataset['train']['sentence']
    train_labels = dataset['train']['label']
    val_texts = dataset['validation']['sentence']
    val_labels = dataset['validation']['label']

    def encode(texts, labels):
        inputs = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long)
        }

    train_data = encode(train_texts, train_labels)
    val_data = encode(val_texts, val_labels)

    return train_data, val_data, len(set(train_labels))


def create_dataloader(encoded_data, batch_size=64):
    """
    Create a DataLoader for the encoded dataset.
    """

    class SimpleDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data['input_ids'])

        def __getitem__(self, idx):
            return {key: val[idx] for key, val in self.data.items()}

    dataset = SimpleDataset(encoded_data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_model(model, train_loader, val_loader, num_labels, num_epochs=3, learning_rate=5e-5, device='cuda'):
    """
    Fine-tune BERT model with custom layer weights.
    """
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_scheduler(
        "linear", optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * num_epochs
    )

    # layer_weights = {}
    # layer_weights = {f'encoder.layer.{i}': 1.0 / (i + 1) for i in range(12)}
    layer_weights = {f'encoder.layer.{i}': 1.0 * (i + 1) for i in range(12)}
    print("Layer weights:", layer_weights)

    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    results = []

    for epoch in tqdm(range(num_epochs), desc="Epoch", position=0, leave=False):
        model.train()
        total_loss = 0
        with tqdm(train_loader, desc="Batch", position=1, leave=False) as batch_bar:
            for batch in batch_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                batch_bar.set_postfix(loss=loss.item())

                # Backpropagation
                loss.backward()
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            for layer, weight in layer_weights.items():
                                if layer in name:
                                    param.grad *= weight

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Evaluate on validation set
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()

                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total
        tqdm.write(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Store results
        results.append({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'val_loss': avg_val_loss,
            'accuracy': accuracy
        })

    print("Training complete!")
    # Output results to log
    print(json.dumps(results))
    # with open('training_log.json', 'w') as log_file:
    #     json.dump(results, log_file)


def main():
    # Load dataset
    print("Loading dataset...")
    train_data, val_data, num_labels = load_finetune_dataset(task_name='sst2', max_length=128)

    # Create DataLoaders
    batch_size = 256
    train_loader = create_dataloader(train_data, batch_size)
    val_loader = create_dataloader(val_data, batch_size)

    # Load pre-trained BERT model
    print("Initializing model...")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

    # Fine-tune the model
    print("Starting training...")
    train_model(model, train_loader, val_loader, num_labels, num_epochs=3, learning_rate=5e-5, device='cuda')

    # Save the fine-tuned model
    print("Saving model...")
    model.save_pretrained("./fine_tuned_bert")
    print("Model saved to ./fine_tuned_bert")


if __name__ == "__main__":
    main()
