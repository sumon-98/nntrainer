#!/usr/bin/env python3

import re
import sys
from collections import defaultdict

def parse_weights_file(filepath):
    """
    Parse the weights file and return a dictionary of layers with their weights.
    """
    layers = {}
    current_layer = None
    layer_weights = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Match layer definition
            layer_match = re.match(r'Layer: ([^(]+) \(Type: ([^)]+)\)', line)
            if layer_match:
                # Save previous layer if exists
                if current_layer:
                    layers[current_layer] = layer_weights
                
                # Start new layer
                current_layer = layer_match.group(1).strip()
                layer_weights = {}
                continue
            
            # Match weight definition
            weight_match = re.match(r'Weight (\d+) \(Name: ([^,]+), Dim: (\d+)\): (.+)', line)
            if weight_match and current_layer:
                weight_index = int(weight_match.group(1))
                weight_name = weight_match.group(2).strip()
                weight_dim = int(weight_match.group(3))
                weight_values_str = weight_match.group(4).strip()
                
                # Parse weight values (taking first few values as samples)
                # In a real implementation, we might want to parse all values
                # For now, we'll just store the string representation
                layer_weights[weight_index] = {
                    'name': weight_name,
                    'dim': weight_dim,
                    'values_str': weight_values_str
                }
        
        # Don't forget the last layer
        if current_layer:
            layers[current_layer] = layer_weights
    
    return layers

def are_weights_changed(weight_before, weight_after, threshold=1e-6):
    """
    Compare two weight strings and determine if they've changed significantly.
    
    Args:
        weight_before: Dictionary containing weight information before training
        weight_after: Dictionary containing weight information after training
        threshold: Minimum difference to consider a change (default: 1e-6)
    
    Returns:
        bool: True if weights changed significantly, False otherwise
    """
    # Extract numerical values from the weight strings
    values_before = weight_before['values_str'].split()
    values_after = weight_after['values_str'].split()
    
    # Compare all values (or as many as we can)
    compare_count = min(len(values_before), len(values_after))
    
    # For performance, we'll check a sample of values rather than all
    # But we'll check more than just 10 - let's check up to 100 values
    # spaced throughout the weight vector
    step = max(1, compare_count // min(100, compare_count))
    
    for i in range(0, compare_count, step):
        try:
            val_before = float(values_before[i])
            val_after = float(values_after[i])
            diff = abs(val_before - val_after)
            if diff > threshold:
                return True
        except (ValueError, IndexError):
            # If we can't parse values, assume they're different
            return True
    
    return False

def categorize_layers(layers_before, layers_after):
    """
    Categorize layers based on whether they changed and whether they are LoRA layers.
    """
    results = {
        'lora_changed': 0,
        'lora_unchanged': 0,
        'non_lora_changed': 0,
        'non_lora_unchanged': 0
    }
    
    # Process each layer
    for layer_name in layers_before:
        if layer_name not in layers_after:
            print(f"Warning: Layer {layer_name} not found in after file")
            continue
        
        weights_before = layers_before[layer_name]
        weights_after = layers_after[layer_name]
        
        # Check if this is a LoRA layer (has loraA or loraB weights)
        is_lora_layer = any(
            'loraA' in weight_info['name'] or 'loraB' in weight_info['name']
            for weight_info in weights_before.values()
        )
        
        # Check if any weights changed
        layer_changed = False
        for weight_index in weights_before:
            if weight_index in weights_after:
                if are_weights_changed(weights_before[weight_index], weights_after[weight_index]):
                    layer_changed = True
                    break
            else:
                layer_changed = True
                break
        
        # Categorize
        if is_lora_layer:
            if layer_changed:
                results['lora_changed'] += 1
            else:
                results['lora_unchanged'] += 1
        else:
            if layer_changed:
                results['non_lora_changed'] += 1
            else:
                results['non_lora_unchanged'] += 1
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare model weights before and after LoRA training")
    parser.add_argument("before_file", help="Path to weights file before training")
    parser.add_argument("after_file", help="Path to weights file after training")
    parser.add_argument("--threshold", type=float, default=1e-6, 
                        help="Threshold for considering weights as changed (default: 1e-6)")
    parser.add_argument("--verbose", action="store_true", 
                        help="Print detailed information about weight changes")
    
    args = parser.parse_args()
    
    print(f"Using threshold: {args.threshold}")
    print("Parsing before training weights...")
    layers_before = parse_weights_file(args.before_file)
    print(f"Found {len(layers_before)} layers in before file")
    
    print("Parsing after training weights...")
    layers_after = parse_weights_file(args.after_file)
    print(f"Found {len(layers_after)} layers in after file")
    
    # Update the are_weights_changed function to use the command line threshold
    def are_weights_changed_with_threshold(weight_before, weight_after):
        return are_weights_changed(weight_before, weight_after, args.threshold)
    
    # Update the categorize_layers function to use the new comparison function
    def categorize_layers_with_threshold(layers_before, layers_after):
        """
        Categorize layers based on whether they changed and whether they are LoRA layers.
        """
        results = {
            'lora_changed': 0,
            'lora_unchanged': 0,
            'non_lora_changed': 0,
            'non_lora_unchanged': 0
        }
        
        # Process each layer
        for layer_name in layers_before:
            if layer_name not in layers_after:
                print(f"Warning: Layer {layer_name} not found in after file")
                continue
            
            weights_before = layers_before[layer_name]
            weights_after = layers_after[layer_name]
            
            # Check if this is a LoRA layer (has loraA or loraB weights)
            is_lora_layer = any(
                'loraA' in weight_info['name'] or 'loraB' in weight_info['name']
                for weight_info in weights_before.values()
            )
            
            # Check if any weights changed
            layer_changed = False
            for weight_index in weights_before:
                if weight_index in weights_after:
                    if are_weights_changed_with_threshold(weights_before[weight_index], weights_after[weight_index]):
                        layer_changed = True
                        if args.verbose:
                            print(f"Layer {layer_name} weight {weight_index} changed")
                        break
                else:
                    layer_changed = True
                    if args.verbose:
                        print(f"Layer {layer_name} weight {weight_index} missing in after file")
                    break
            
            # Categorize
            if is_lora_layer:
                if layer_changed:
                    results['lora_changed'] += 1
                else:
                    results['lora_unchanged'] += 1
            else:
                if layer_changed:
                    results['non_lora_changed'] += 1
                else:
                    results['non_lora_unchanged'] += 1
        
        return results
    
    print("Comparing weights...")
    results = categorize_layers_with_threshold(layers_before, layers_after)
    
    print("\n=== Comparison Results ===")
    print(f"LoRA layers that changed: {results['lora_changed']}")
    print(f"LoRA layers that didn't change: {results['lora_unchanged']}")
    print(f"Non-LoRA layers that changed: {results['non_lora_changed']}")
    print(f"Non-LoRA layers that didn't change: {results['non_lora_unchanged']}")
    
    # Additional statistics
    total_lora_layers = results['lora_changed'] + results['lora_unchanged']
    total_non_lora_layers = results['non_lora_changed'] + results['non_lora_unchanged']
    
    print(f"\n=== Summary ===")
    print(f"Total LoRA layers: {total_lora_layers}")
    print(f"Total non-LoRA layers: {total_non_lora_layers}")
    if total_lora_layers > 0:
        print(f"Percentage of LoRA layers that changed: {results['lora_changed']/total_lora_layers*100:.2f}%")
    else:
        print("Percentage of LoRA layers that changed: 0.00%")
    if total_non_lora_layers > 0:
        print(f"Percentage of non-LoRA layers that changed: {results['non_lora_changed']/total_non_lora_layers*100:.2f}%")
    else:
        print("Percentage of non-LoRA layers that changed: 0.00%")

if __name__ == "__main__":
    main()
