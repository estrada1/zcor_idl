"""
Probabilistic Finite State Automata (PFSA) for ZCoR-IPF.

Implements the PFSA inference and Sequence Likelihood Defect (SLD) computation
described in the paper (Methods, Step 2).

A PFSA is a specialized HMM that models categorical stochastic processes.
For ZCoR-IPF:
- Alphabet: {0, 1, 2} (trinary encoding)
- 204 PFSAs total: 51 categories × 2 cohorts (positive/control) × 2 sexes

The SLD Δ measures divergence between positive and control models:
  Δ = L(G_control, x) - L(G_positive, x)

Reference: Chattopadhyay & Lipson (2013)
"""

import numpy as np
from collections import defaultdict, Counter
import warnings
import time
import sys
import math


class PFSA:
    """
    Probabilistic Finite State Automaton for trinary sequences.

    Uses variable-order Markov chain estimation. Transition probabilities are
    stored in a flat lookup table indexed by integer context encoding for fast
    vectorized scoring.
    """

    ALPHABET_SIZE = 3  # {0, 1, 2}

    def __init__(self, max_depth=3, smoothing=1e-6):
        self.max_depth = max_depth
        self.smoothing = smoothing
        # After fit: log_prob_table[context_int, symbol] = log P(symbol|context)
        # context_int encodes a fixed-length context as base-4 number
        # (0=no-symbol-sentinel, 1/2/3 = alphabet 0/1/2)
        self._log_prob_tables = {}  # depth -> np.array of shape (4**depth, 3)
        self._default_log_probs = np.log(np.ones(self.ALPHABET_SIZE) / self.ALPHABET_SIZE)
        self.is_fitted = False

    @staticmethod
    def _context_to_int(context, depth):
        """Encode a context tuple as an integer. Each position uses base-4."""
        val = 0
        for c in context[-depth:] if len(context) >= depth else context:
            val = val * 4 + (int(c) + 1)
        # Pad with zeros if context is shorter than depth
        pad = depth - min(len(context), depth)
        val = val  # leading zeros are implicit
        return val

    def fit(self, sequences):
        """
        Learn transition probabilities from sequences.
        Args:
            sequences: array of shape (n_sequences, seq_length) with values {0,1,2}
        """
        sequences = np.asarray(sequences, dtype=np.int8)
        n_seq, seq_len = sequences.shape

        # Build count tables for each depth
        for depth in range(self.max_depth + 1):
            n_contexts = 4 ** depth if depth > 0 else 1
            counts = np.zeros((n_contexts, self.ALPHABET_SIZE), dtype=np.float64)

            if depth == 0:
                # No context: just count symbol frequencies
                for s in range(self.ALPHABET_SIZE):
                    counts[0, s] = np.sum(sequences == s)
            else:
                # Vectorized context encoding using sliding windows
                for seq in sequences:
                    for i in range(depth, seq_len):
                        ctx = 0
                        for d in range(depth):
                            ctx = ctx * 4 + (int(seq[i - depth + d]) + 1)
                        counts[ctx, int(seq[i])] += 1

            # Convert to log-probabilities with smoothing
            total = counts.sum(axis=1, keepdims=True) + self.ALPHABET_SIZE * self.smoothing
            probs = (counts + self.smoothing) / total
            self._log_prob_tables[depth] = np.log(probs)

        self.is_fitted = True

    def batch_log_likelihood(self, sequences):
        """
        Compute negative normalized log-likelihood for a batch of sequences.

        Much faster than calling log_likelihood() per sequence.

        Args:
            sequences: array of shape (n_sequences, seq_length)

        Returns:
            array of shape (n_sequences,) with negative normalized log-likelihoods
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")

        sequences = np.asarray(sequences, dtype=np.int8)
        n_seq, seq_len = sequences.shape
        depth = self.max_depth

        # Use the deepest available context table
        log_table = self._log_prob_tables[depth]

        # Precompute context integers for all positions in all sequences
        # For positions < depth, use shorter contexts (fall back to depth-0)
        log_probs = np.zeros(n_seq, dtype=np.float64)

        # Handle early positions (context shorter than max_depth) with depth-0
        for i in range(min(depth, seq_len)):
            symbols = sequences[:, i]
            # Use unigram (depth-0) for early positions
            log_probs += self._log_prob_tables[0][0, symbols]

        # Handle remaining positions with full-depth context
        if seq_len > depth:
            # Build context integers for all (sequence, position) pairs
            # Context at position i = seq[i-depth:i], encoded as base-4 integer
            powers = 4 ** np.arange(depth - 1, -1, -1)  # [4^(d-1), ..., 4^0]

            for i in range(depth, seq_len):
                # context values: seq[:, i-depth:i] + 1 (shift 0,1,2 → 1,2,3)
                ctx_slice = sequences[:, i - depth:i].astype(np.int32) + 1
                ctx_ints = ctx_slice @ powers
                symbols = sequences[:, i]
                log_probs += log_table[ctx_ints, symbols]

        return -log_probs / max(seq_len, 1)


class PFSAEnsemble:
    """Collection of 204 PFSA models: 51 categories × 2 cohorts × 2 sexes."""

    def __init__(self, n_categories=51, max_depth=3, smoothing=1e-6):
        self.n_categories = n_categories
        self.max_depth = max_depth
        self.smoothing = smoothing
        self.models = {
            sex: {
                cohort: [
                    PFSA(max_depth=max_depth, smoothing=smoothing)
                    for _ in range(n_categories)
                ]
                for cohort in ["positive", "control"]
            }
            for sex in ["M", "F"]
        }

    def fit(self, series_array, metadata):
        """Train all 204 PFSA models."""
        indices = {"M": {"positive": [], "control": []},
                   "F": {"positive": [], "control": []}}
        for i, meta in enumerate(metadata):
            sex = meta["sex"]
            cohort = "positive" if meta["label"] == 1 else "control"
            indices[sex][cohort].append(i)

        total = 0
        for sex in ["M", "F"]:
            for cohort in ["positive", "control"]:
                idx = indices[sex][cohort]
                n = len(idx)
                if n == 0:
                    warnings.warn(f"No patients for {sex}/{cohort}")
                    continue
                t0 = time.time()
                for cat in range(self.n_categories):
                    cat_sequences = series_array[idx, cat, :]
                    self.models[sex][cohort][cat].fit(cat_sequences)
                    total += 1
                elapsed = time.time() - t0
                print(f"  {sex}/{cohort}: {n} patients, "
                      f"{self.n_categories} models in {elapsed:.1f}s")

        print(f"  Trained {total} PFSA models total")

    def compute_sld(self, series_array, metadata):
        """
        Compute SLD for all patients, vectorized per-category.

        Returns (sld, neg_llk, pos_llk), each shape (n_patients, n_categories).
        """
        n_patients = len(metadata)
        sld = np.zeros((n_patients, self.n_categories))
        neg_llk = np.zeros((n_patients, self.n_categories))
        pos_llk = np.zeros((n_patients, self.n_categories))

        # Process per sex to use batch scoring
        for sex in ["M", "F"]:
            sex_idx = [i for i, m in enumerate(metadata) if m["sex"] == sex]
            if not sex_idx:
                continue
            sex_idx = np.array(sex_idx)
            n_sex = len(sex_idx)

            t0 = time.time()
            for cat in range(self.n_categories):
                seqs = series_array[sex_idx, cat, :]  # (n_sex, n_weeks)

                ctrl_llk = self.models[sex]["control"][cat].batch_log_likelihood(seqs)
                pos_l = self.models[sex]["positive"][cat].batch_log_likelihood(seqs)

                neg_llk[sex_idx, cat] = ctrl_llk
                pos_llk[sex_idx, cat] = pos_l
                sld[sex_idx, cat] = ctrl_llk - pos_l

                # Progress every 10 categories
                if (cat + 1) % 10 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / (cat + 1) * (self.n_categories - cat - 1)
                    print(f"    {sex}: {cat+1}/{self.n_categories} categories "
                          f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

            elapsed = time.time() - t0
            print(f"    {sex}: done ({n_sex} patients, {elapsed:.1f}s)")

        return sld, neg_llk, pos_llk


def compute_pfsa_features(series_array, metadata, max_depth=3, smoothing=1e-6):
    """End-to-end: train PFSAs and compute SLD features."""
    print("Training PFSA models...")
    ensemble = PFSAEnsemble(
        n_categories=series_array.shape[1],
        max_depth=max_depth,
        smoothing=smoothing,
    )
    ensemble.fit(series_array, metadata)

    print("\nComputing SLD features...")
    sld, neg_llk, pos_llk = ensemble.compute_sld(series_array, metadata)

    return ensemble, sld, neg_llk, pos_llk


if __name__ == "__main__":
    import os
    import csv
    import argparse

    parser = argparse.ArgumentParser(description="Train PFSAs and compute SLD features")
    parser.add_argument("--subset", type=int, default=0,
                        help="Use only N patients (0=all). Useful for fast iteration.")
    parser.add_argument("--max-depth", type=int, default=3,
                        help="PFSA context depth (default: 3)")
    args = parser.parse_args()

    encoded_dir = os.path.join("data", "processed", "encoded")
    print("Loading encoded data...")
    data = np.load(os.path.join(encoded_dir, "trinary_series.npz"))
    series = data["series"]

    with open(os.path.join(encoded_dir, "patient_metadata.csv")) as f:
        metadata = list(csv.DictReader(f))
        for m in metadata:
            m["label"] = int(m["label"])

    # Subset if requested — keep all positives, subsample controls
    if args.subset > 0 and args.subset < len(metadata):
        pos_idx = [i for i, m in enumerate(metadata) if m["label"] == 1]
        ctrl_idx = [i for i, m in enumerate(metadata) if m["label"] == 0]
        n_ctrl = max(args.subset - len(pos_idx), 0)
        rng = np.random.default_rng(42)
        ctrl_sample = rng.choice(ctrl_idx, size=min(n_ctrl, len(ctrl_idx)), replace=False)
        keep = sorted(list(pos_idx) + list(ctrl_sample))
        series = series[keep]
        metadata = [metadata[i] for i in keep]
        print(f"  Subset: using {len(metadata)} patients")

    n_pos = sum(1 for m in metadata if m["label"] == 1)
    print(f"  Series shape: {series.shape}")
    print(f"  Positive: {n_pos}, Control: {len(metadata) - n_pos}")

    t_start = time.time()
    ensemble, sld, neg_llk, pos_llk = compute_pfsa_features(
        series, metadata, max_depth=args.max_depth
    )
    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s")

    # Save
    output_dir = os.path.join("data", "processed", "features")
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(output_dir, "pfsa_features.npz"),
        sld=sld, neg_llk=neg_llk, pos_llk=pos_llk,
    )
    # Also save the metadata used (in case subset was applied)
    with open(os.path.join(output_dir, "pfsa_metadata.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)
    print(f"Saved PFSA features to {output_dir}/")

    # Summary
    print(f"\nSLD statistics:")
    print(f"  Shape: {sld.shape}")
    print(f"  Mean:  {np.mean(sld):.4f}")
    print(f"  Std:   {np.std(sld):.4f}")

    pos_mask = np.array([m["label"] == 1 for m in metadata])
    ctrl_mask = ~pos_mask
    pos_mean = np.mean(sld[pos_mask]) if pos_mask.any() else 0
    ctrl_mean = np.mean(sld[ctrl_mask]) if ctrl_mask.any() else 0
    print(f"\n  Mean SLD (positive): {pos_mean:.4f}")
    print(f"  Mean SLD (control):  {ctrl_mean:.4f}")
    print(f"  Separation:          {pos_mean - ctrl_mean:.4f}")
