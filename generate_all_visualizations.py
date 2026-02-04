"""
GENERATE ALL VISUALIZATIONS
============================

This script runs all algorithms and generates their story-telling visualizations.

Each visualization is designed to tell a complete story:
1. What the algorithm does (core mechanism)
2. What it learns (parameters, features, patterns)
3. When it fails (inductive bias limits)
4. How hyperparameters affect it (ablations visualized)
5. Comparison to alternatives (relative strengths)
"""

import subprocess
import os
import sys

# Files that need visualizations generated
ALGORITHMS = [
    ('13_cnn.py', 'CNN — Local patterns + Weight sharing'),
    ('14_rnn_lstm.py', 'RNN/LSTM — Sequential memory + Gates'),
    ('15_transformer.py', 'Transformer — Attention patterns'),
    ('16_gmm.py', 'GMM — Mixture model + EM'),
    ('17_hmm.py', 'HMM — Hidden states + Decoding'),
    ('18_vae.py', 'VAE — Latent space + Generation'),
    ('23_maml.py', 'MAML — Meta-learning + Adaptation'),
]

def run_algorithm(filename, description):
    """Run an algorithm file and wait for completion."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"File: {filename}")
    print('='*60)

    try:
        result = subprocess.run(
            [sys.executable, filename],
            cwd='/Users/sid47/ML Algorithms',
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per file
        )

        if result.returncode == 0:
            print(f"✓ {filename} completed successfully")
            # Check if PNG was created
            png_base = filename.replace('.py', '')
            pngs = [f for f in os.listdir('/Users/sid47/ML Algorithms')
                   if f.startswith(png_base) and f.endswith('.png')]
            if pngs:
                print(f"  Generated: {', '.join(pngs)}")
            return True
        else:
            print(f"✗ {filename} failed")
            print(f"  Error: {result.stderr[:500]}")
            return False

    except subprocess.TimeoutExpired:
        print(f"✗ {filename} timed out (>10 min)")
        return False
    except Exception as e:
        print(f"✗ {filename} error: {e}")
        return False

def main():
    print("="*60)
    print("GENERATING ALL VISUALIZATIONS")
    print("="*60)

    os.chdir('/Users/sid47/ML Algorithms')

    results = {}
    for filename, description in ALGORITHMS:
        results[filename] = run_algorithm(filename, description)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for filename, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {filename}")

    # List all generated PNGs
    print("\nGenerated visualizations:")
    for filename, _ in ALGORITHMS:
        base = filename.replace('.py', '')
        pngs = [f for f in os.listdir('.') if f.startswith(base) and f.endswith('.png')]
        for png in pngs:
            size = os.path.getsize(png) // 1024
            print(f"  {png} ({size} KB)")

if __name__ == '__main__':
    main()
