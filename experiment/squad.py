import json
import torch
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizerFast,
    BertForQuestionAnswering,
    AdamW,
    get_scheduler,
    default_data_collator
)
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from evaluate import load
from cal_weight import calculate_all_differences, simple_reweight, baseline_weight


def load_finetune_dataset(max_length=384, doc_stride=128):
    """
    Load and preprocess the SQuAD dataset.
    """
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Load SQuAD dataset
    dataset = load_dataset("squad")

    def prepare_train_features(examples):
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")

        start_positions = []
        end_positions = []
        answers_list = []  # List to store ground-truth answers for evaluation

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            if len(answers["answer_start"]) == 0:
                # No answer case: Set start and end positions to [CLS] token
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                answers_list.append(answers)  # Append an empty string for no-answer case
            else:
                # Start and end character positions of the answer
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Find the token indices corresponding to the answer
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Check if answer is out of bounds (truncated)
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    start_positions.append(cls_index)
                    end_positions.append(cls_index)
                    answers_list.append(answers)  # Append empty string for out-of-bounds case
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    start_positions.append(token_start_index - 1)

                    while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    end_positions.append(token_end_index + 1)

                    answers_list.append(answers)

        tokenized_examples["start_positions"] = start_positions
        tokenized_examples["end_positions"] = end_positions
        tokenized_examples["answers"] = answers_list  # Add answers to the output
        return tokenized_examples

    # Preprocess train and validation datasets
    train_data = dataset["train"].map(
        prepare_train_features,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    val_data = dataset["validation"].map(
        prepare_train_features,
        batched=True,
        remove_columns=dataset["validation"].column_names
    )

    return train_data, val_data


def custom_collate_fn(features):
    """
    Custom collate function to include 'answers' in the batch.
    """

    # Add the 'answers' field explicitly
    if "answers" in features[0]:
        tmp = [feature["answers"] for feature in features]
    for feat in features:
        del feat["answers"]

    # Use the default collator for tensorizing fields like input_ids, attention_mask, etc.
    batch = default_data_collator(features)
    batch["answers"] = tmp

    return batch

def create_dataloader(dataset, batch_size=16):
    """
    Create a DataLoader for the encoded QA dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn  # Use the default data collator
    )


def train_model(model, train_loader, val_loader, num_epochs=3, learning_rate=5e-5, device='cuda'):
    """
    Fine-tune BERT model for Question Answering.
    """
    squad_metric = load("squad")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_scheduler(
        "linear", optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * num_epochs
    )

    train_loss_df = pd.DataFrame(columns=["epoch", "batch", "loss", "lr"])
    eval_loss_df = pd.DataFrame(columns=["epoch", "loss", "EM", "F1"])

    model.to(device)
    print(model)
    num_heads = 12
    head_size = 64  # Hidden size divided by number of heads
    # Layer and head-specific weights
    layer_head_weights = simple_reweight("bert-base-uncased", "Intel/bert-base-uncased-squadv1.1-sparse-80-1x4-block-pruneofa")

    # print("Using zero-initialized layer-head weights for encoder layers.")
    print("Weights for encoder layers: {}".format(layer_head_weights))

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        total_loss = 0

        with tqdm(train_loader, desc="Training", leave=False) as batch_bar:
            for batch in batch_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)

                optimizer.zero_grad()

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions,
                )
                loss = outputs.loss
                total_loss += loss.item()
                batch_bar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
                train_loss_df = pd.concat([train_loss_df, pd.DataFrame([{
                    "epoch": epoch + 1,
                    "batch": batch_bar.n,
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"]
                }])], ignore_index=True)

                # Backward pass
                loss.backward()
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            # If the parameter is specific to encoder.layer heads
                            if "encoder.layer" in name and ("query" in name or "key" in name or "value" in name):
                                # Split the gradient into individual heads
                                grad = param.grad.view(num_heads, head_size,
                                                       -1)  # Reshape to (num_heads, head_size, hidden_size)

                                # Apply head-specific weights
                                layer_idx = int(name.split(".")[3])  # Extract layer index from the parameter name
                                for head_idx in range(num_heads):
                                    weight_key = f"bert.encoder.layer.{layer_idx}.head.{head_idx}"  # Generate the key
                                    if weight_key in layer_head_weights:
                                        grad[head_idx] *= layer_head_weights[weight_key]
                                    else:
                                        print(f"Weight for {weight_key} not found in layer_head_weights.")

                                # Reshape back to the original size
                                param.grad = grad.view(param.grad.shape)
                            elif name in layer_head_weights:
                                # For non-encoder parts, directly apply the weight from layer_head_weights
                                param.grad *= layer_head_weights[name]
                            else:
                                print(f"Weight for {name} not found in layer_head_weights.")

                    # Optimizer and scheduler step
                    optimizer.step()
                    scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        predicted_texts = []
        ground_truth_texts = []
        total_val_loss = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)

                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions,
                )

                # Accumulate validation loss
                total_val_loss += outputs.loss.item()

                # Get predictions
                predicted_start = torch.argmax(outputs.start_logits, dim=1)
                predicted_end = torch.argmax(outputs.end_logits, dim=1)

                # Convert token indices to text
                for i, (start, end) in enumerate(zip(predicted_start, predicted_end)):
                    input_tokens = input_ids[i]
                    predicted_tokens = tokenizer.convert_ids_to_tokens(input_tokens[start:end + 1])
                    predicted_texts.append(tokenizer.convert_tokens_to_string(predicted_tokens))
                    ground_truth_texts.append(batch["answers"][i])  # Assuming single ground truth

        # Calculate average validation loss
        avg_val_loss = total_val_loss / len(val_loader)

        # Prepare references and predictions for metrics
        references = [{"id": str(i), "answers": gt} for i, gt in enumerate(ground_truth_texts)]
        predictions = [{"id": str(i), "prediction_text": pred} for i, pred in enumerate(predicted_texts)]

        # Calculate EM and F1 metrics
        results = squad_metric.compute(predictions=predictions, references=references)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Exact Match (EM): {results['exact_match']:.2f}")
        print(f"F1 Score: {results['f1']:.2f}")

        eval_loss_df = pd.concat([eval_loss_df, pd.DataFrame([{
            "epoch": epoch + 1,

            "loss": avg_val_loss,
            "EM": results["exact_match"],
            "F1": results["f1"]
        }])], ignore_index=True)

    print("Training complete!")
    train_loss_df.to_csv("train_loss.csv", index=False)
    eval_loss_df.to_csv("eval_loss.csv", index=False)


def main():
    print("Loading dataset...")
    train_data, val_data = load_finetune_dataset()

    print("Creating DataLoaders...")
    train_loader = create_dataloader(train_data, batch_size=64)
    val_loader = create_dataloader(val_data, batch_size=32)

    print("Initializing model...")
    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

    print("Starting training...")
    train_model(model, train_loader, val_loader, num_epochs=3, learning_rate=5e-5, device="cuda")

    print("Saving model...")
    model.save_pretrained("./fine_tuned_bert_qa")
    print("Model saved to ./fine_tuned_bert_qa")


if __name__ == "__main__":
    main()
