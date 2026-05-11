#!/usr/bin/env python3
"""
Compare NNTrainer model weight dumps before and after training.

Works for both LoRA and normal (full fine-tuning) training.
For LoRA: verifies that only loraA/loraB weights changed, base weights frozen.
For Normal: verifies that all trainable weights changed.

Usage:
    python3 compare_weights.py model_weights_before_training.txt model_weights_after_training.txt
    python3 compare_weights.py before.txt after.txt --mode lora --verbose
    python3 compare_weights.py before.txt after.txt --mode normal --verbose
"""

import re
import sys
import argparse
from collections import OrderedDict

# --- Parsing ---

def parse_weights_file(filepath):
    """
    Parse the weight dump text file.
    Returns: OrderedDict of { layer_name: { weight_index: {name, dim, values} } }
    """
    layers = OrderedDict()
    current_layer = None
    current_type = None
    current_weights = OrderedDict()

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Match layer header: "Layer: layer0_wq (Type: fully_connected)"
            layer_match = re.match(r'^Layer:\s+(.+?)\s+\(Type:\s+(.+?)\)$', line)
            if layer_match:
                if current_layer is not None:
                    layers[current_layer] = {
                        'type': current_type,
                        'weights': current_weights
                    }
                current_layer = layer_match.group(1).strip()
                current_type = layer_match.group(2).strip()
                current_weights = OrderedDict()
                continue

            # Match weight line: "  Weight 0 (Name: layer0_wq:weight, Dim: 2097152): 0.03 0.02 ..."
            weight_match = re.match(
                r'^\s*Weight\s+(\d+)\s+\(Name:\s+(.+?),\s+Dim:\s+(\d+)\):\s+(.+)$', line
            )
            if weight_match and current_layer is not None:
                idx = int(weight_match.group(1))
                name = weight_match.group(2).strip()
                dim = int(weight_match.group(3))
                raw = weight_match.group(4).strip()

                # Parse all visible float values (skip "...")
                values = []
                for tok in raw.split():
                    if tok == '...':
                        continue
                    try:
                        values.append(float(tok))
                    except ValueError:
                        pass

                current_weights[idx] = {
                    'name': name,
                    'dim': dim,
                    'values': values
                }

    # Don't forget the last layer
    if current_layer is not None:
        layers[current_layer] = {
            'type': current_type,
            'weights': current_weights
        }

    return layers

# --- Comparison ---

def compare_weight_values(vals_before, vals_after, threshold=1e-6):
    """
    Compare two lists of float values. Returns (changed: bool, max_diff: float).
    """
    if len(vals_before) != len(vals_after):
        return True, float('inf')

    max_diff = 0.0
    for a, b in zip(vals_before, vals_after):
        d = abs(a - b)
        if d > max_diff:
            max_diff = d

    return max_diff > threshold, max_diff

def classify_weight_name(wname):
    """Classify a weight by its suffix."""
    if ':loraA' in wname:
        return 'loraA'
    elif ':loraB' in wname:
        return 'loraB'
    elif ':weight' in wname:
        return 'base_weight'
    elif ':bias' in wname:
        return 'bias'
    elif ':gamma' in wname:
        return 'gamma'
    elif ':Embedding' in wname:
        return 'embedding'
    else:
        return 'other'

# --- Main ---

def main():
    parser = argparse.ArgumentParser(
        description="Compare NNTrainer weight dumps before/after training"
    )
    parser.add_argument("before_file", help="Weights file before training")
    parser.add_argument("after_file", help="Weights file after training")
    parser.add_argument("--threshold", type=float, default=1e-6,
                        help="Min diff to consider as changed (default: 1e-6)")
    parser.add_argument("--mode", choices=["lora", "normal", "auto"], default="auto",
                        help="Training mode: lora, normal, or auto-detect (default: auto)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-weight detailed comparison")
    args = parser.parse_args()

    print(f"Threshold: {args.threshold}")
    print(f"Parsing: {args.before_file}")
    layers_before = parse_weights_file(args.before_file)
    print(f"  Found {len(layers_before)} layers")

    print(f"Parsing: {args.after_file}")
    layers_after = parse_weights_file(args.after_file)
    print(f"  Found {len(layers_after)} layers")

    # Auto-detect mode: if any weight name contains "loraA", it's LoRA mode
    if args.mode == "auto":
        has_lora = False
        for ldata in layers_before.values():
            for wdata in ldata['weights'].values():
                if 'loraA' in wdata['name'] or 'loraB' in wdata['name']:
                    has_lora = True
                    break
            if has_lora:
                break
        args.mode = "lora" if has_lora else "normal"
        print(f"Auto-detected mode: {args.mode}")

    # Per-weight results
    results = []  # list of (layer_name, layer_type, weight_name, weight_class, changed, max_diff)

    for layer_name, ldata_before in layers_before.items():
        if layer_name not in layers_after:
            print(f"  WARNING: Layer '{layer_name}' missing in after file")
            continue

        ldata_after = layers_after[layer_name]

        for widx, wbefore in ldata_before['weights'].items():
            if widx not in ldata_after['weights']:
                print(f"  WARNING: Weight {widx} of '{layer_name}' missing in after file")
                continue

            wafter = ldata_after['weights'][widx]
            wclass = classify_weight_name(wbefore['name'])
            changed, max_diff = compare_weight_values(
                wbefore['values'], wafter['values'], args.threshold
            )
            results.append((
                layer_name, ldata_before['type'],
                wbefore['name'], wclass, changed, max_diff
            ))

    # --- Print results ---
    print("\n" + "=" * 90)
    print("  WEIGHT COMPARISON RESULTS")
    print("=" * 90)

    if args.verbose:
        print(f"\n{'Weight Name':<55} {'Class':<14} {'Changed?':<10} {'MaxDiff':<12}")
        print("-" * 90)
        for layer_name, ltype, wname, wclass, changed, max_diff in results:
            status = "CHANGED" if changed else "FROZEN"
            diff_str = f"{max_diff:.8f}" if max_diff != float('inf') else "N/A"
            print(f"  {wname:<53} {wclass:<14} {status:<10} {diff_str:<12}")

    # --- Aggregate by class ---
    class_stats = {}
    for _, _, wname, wclass, changed, _ in results:
        if wclass not in class_stats:
            class_stats[wclass] = {'total': 0, 'changed': 0, 'frozen': 0}
        class_stats[wclass]['total'] += 1
        if changed:
            class_stats[wclass]['changed'] += 1
        else:
            class_stats[wclass]['frozen'] += 1

    print(f"\n{'='*60}")
    print(f"  SUMMARY BY WEIGHT CLASS")
    print(f"{'='*60}")
    print(f"  {'Class':<16} {'Total':<8} {'Changed':<10} {'Frozen':<10}")
    print(f"  {'-'*44}")
    for wclass in ['base_weight', 'bias', 'loraA', 'loraB', 'gamma', 'embedding', 'other']:
        if wclass in class_stats:
            s = class_stats[wclass]
            print(f"  {wclass:<16} {s['total']:<8} {s['changed']:<10} {s['frozen']:<10}")

    # --- Verdict ---
    print(f"\n{'='*60}")
    print(f"  VERDICT (mode={args.mode})")
    print(f"{'='*60}")

    all_ok = True
    if args.mode == "lora":
        # LoRA expectations:
        # - loraA, loraB: SHOULD change
        # - base_weight, bias, gamma, embedding: should NOT change
        for wclass in ['base_weight', 'bias', 'gamma', 'embedding']:
            if wclass in class_stats and class_stats[wclass]['changed'] > 0:
                print(f"  ❌ FAIL: {class_stats[wclass]['changed']} '{wclass}' weights CHANGED (should be frozen)")
                all_ok = False
            elif wclass in class_stats:
                print(f"  ✅ PASS: All {class_stats[wclass]['total']} '{wclass}' weights are FROZEN")

        for wclass in ['loraA', 'loraB']:
            if wclass in class_stats and class_stats[wclass]['frozen'] > 0:
                print(f"  ⚠️  WARN: {class_stats[wclass]['frozen']} '{wclass}' weights did NOT change")
            elif wclass in class_stats:
                print(f"  ✅ PASS: All {class_stats[wclass]['total']} '{wclass}' weights CHANGED (trained)")

    elif args.mode == "normal":
        # Normal training expectations:
        # - base_weight, bias, gamma: SHOULD change
        # - No LoRA weights expected
        for wclass in ['base_weight', 'bias', 'gamma']:
            if wclass in class_stats and class_stats[wclass]['frozen'] > 0:
                print(f"  ⚠️  WARN: {class_stats[wclass]['frozen']}/{class_stats[wclass]['total']} '{wclass}' weights did NOT change")
            elif wclass in class_stats:
                print(f"  ✅ PASS: All {class_stats[wclass]['total']} '{wclass}' weights CHANGED (trained)")

    if all_ok:
        print(f"\n  🎉 All checks passed for {args.mode} mode!")
    else:
        print(f"\n  ⛔ Some checks FAILED — review the details above")

    print()

if __name__ == "__main__":
    main()
