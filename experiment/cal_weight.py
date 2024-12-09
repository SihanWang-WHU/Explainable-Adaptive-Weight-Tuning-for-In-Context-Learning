import torch
from transformers import BertModel


def baseline_weight(pretrained_model_name, checkpoint_path):
    layer_head_weights = calculate_all_differences(pretrained_model_name, checkpoint_path)
    layer_head_weights = {k: torch.tensor(1.0).to('cuda') for k, v in layer_head_weights.items()}
    return layer_head_weights


def simple_reweight(pretrained_model_name, checkpoint_path):
    num_layers = 12
    num_heads = 12

    # Create a dictionary to store layer-head differences
    normalized_differences = calculate_all_differences(pretrained_model_name, checkpoint_path)
    normalized_differences = {k: torch.tensor(1.0).to('cuda') for k, v in normalized_differences.items()}
    for layer in range(num_layers):
        for head in range(num_heads):
            normalized_differences[f"bert.encoder.layer.{layer}.head.{head}"] = torch.tensor(1.0).to('cuda') * (
                        layer + 1)
    return normalized_differences


def calculate_all_differences(pretrained_model_name, checkpoint_path):
    # Load pretrained model
    pretrained_model = BertModel.from_pretrained(pretrained_model_name)

    # Load model from checkpoint
    checkpoint_model = BertModel.from_pretrained(checkpoint_path)

    # Verify both models have the same parameter structure
    assert list(pretrained_model.state_dict().keys()) == list(checkpoint_model.state_dict().keys()), \
        "Pretrained and checkpoint models must have the same structure."

    # Initialize results
    num_layers = 12
    num_heads = 12
    head_size = 64  # Hidden size (768) divided by number of heads (12)

    # Create a dictionary to store layer-head differences
    layer_head_differences = {}

    # Iterate through all model parameters
    for name, param in pretrained_model.named_parameters():
        if name in checkpoint_model.state_dict():
            # Get the corresponding parameter from the checkpoint
            checkpoint_param = checkpoint_model.state_dict()[name]

            # Calculate the absolute difference
            abs_diff = torch.abs(param - checkpoint_param)

            # Process encoder layers with heads
            if "encoder.layer" in name and ("query" in name or "key" in name or "value" in name):
                # Extract layer and head information
                layer_idx = int(name.split(".")[2])
                head_idx = None
                if "query" in name or "key" in name or "value" in name:
                    # Split into heads
                    reshaped_diff = abs_diff.view(num_heads, head_size, -1)
                    for head_idx in range(num_heads):
                        key = f"encoder.layer.{layer_idx}.head.{head_idx}"
                        if key not in layer_head_differences:
                            layer_head_differences[key] = 0.0
                        layer_head_differences[key] += reshaped_diff[head_idx].mean().item()

            # For non-encoder parts, store differences directly
            else:
                if name not in layer_head_differences:
                    layer_head_differences[name] = 0.0
                layer_head_differences[name] += abs_diff.mean().item()
        else:
            print(f"Parameter {name} not found in checkpoint model.")

    # Normalize encoder differences by 3 (query, key, value)
    for key in layer_head_differences:
        if "encoder.layer" in key:
            layer_head_differences[key] /= 3

    # Calculate the sum of all weights and normalize them to have a mean of 1
    total_sum = sum(layer_head_differences.values()) / len(layer_head_differences)
    normalized_differences = {'bert.' + k: torch.tensor(v / total_sum).to('cuda') for k, v in
                              layer_head_differences.items()}
    normalized_differences['qa_outputs.weight'] = torch.tensor(2.0).to('cuda')
    normalized_differences['qa_outputs.bias'] = torch.tensor(2.0).to('cuda')

    # Print results with 3 decimal precision
    print("Layer-Head Absolute Differences (Averaged):")
    for key, value in normalized_differences.items():
        print(f"{key}: {value:.3f}")

    return normalized_differences


if __name__ == "__main__":
    # Example usage
    pretrained_model_name = "bert-base-uncased"  # Replace with your pretrained model name
    checkpoint_path = "fine_tuned_bert_qa"  # Replace with your checkpoint path
    calculate_all_differences(pretrained_model_name, checkpoint_path)
