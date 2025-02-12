#!/usr/bin/env python
import argparse
import numpy as np
from gensim.models import KeyedVectors

def convert_keys_to_int(input_path, output_path):
    """
    Load a KeyedVectors model from input_path, convert all keys to int (via float conversion so that
    keys like '137773608.0' and '137773608' become the same), and handle duplicates by keeping only the first entry.
    Save the new KeyedVectors model to output_path.
    """
    # Load the existing KeyedVectors model.
    kv = KeyedVectors.load(input_path, mmap='r')
    print("Loaded KeyedVectors model.")

    new_keys = []       # List for new keys of type int
    new_vectors = []    # List for corresponding vectors
    seen_keys = set()   # To keep track of which keys (int) have been seen

    # Iterate over keys in the order defined by kv.index_to_key
    for key in kv.index_to_key:
        try:
            # Convert key first to float and then to int. This ensures that '137773608.0'
            # and '137773608' yield the same int (137773608)
            int_key = int(float(key))
        except ValueError:
            # If conversion fails, skip this key.
            continue

        # Skip duplicate keys
        if int_key in seen_keys:
            continue

        seen_keys.add(int_key)
        new_keys.append(int_key)
        # Get vector by using key_to_index mapping.
        vector = kv.vectors[kv.key_to_index[key]]
        new_vectors.append(vector)

    if not new_vectors:
        raise ValueError("No keys could be converted to int.")

    new_vectors = np.array(new_vectors)
    print(f"Converted {len(new_keys)} keys to int.")

    # Create a new KeyedVectors instance with the same vector size
    new_kv = KeyedVectors(vector_size=kv.vector_size)
    # Use the add_vectors method to add new keys and their corresponding vectors.
    new_kv.add_vectors(new_keys, new_vectors)
    print("Created new KeyedVectors model with uniform int keys.")

    # Save the new KeyedVectors model to disk.
    new_kv.save(output_path)
    print(f"Saved new KeyedVectors model to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert all keys in a gensim KeyedVectors model to int (keys are first converted via float conversion) and remove duplicates."
    )
    parser.add_argument("input", help="Path to the input KeyedVectors model file")
    parser.add_argument("output", help="Path to save the new KeyedVectors model with int keys")
    args = parser.parse_args()
    convert_keys_to_int(args.input, args.output)

if __name__ == "__main__":
    main()