import numpy as np
from tqdm import tqdm
from typing import Tuple
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import logging
from utils.logging import setup_logger


def viterbi(states_set: np.ndarray,
            transition_matrix: np.ndarray,
            emission_matrix: np.ndarray,
            pi: np.ndarray,
            observations: np.ndarray,
            tqdm_disable: bool=False, **kwargs):
    """
    Implements the Viterbi algorithm to find the most probable hidden state sequence 
    given a sequence of observations.

    Parameters:
        states_set:         Array of state indices (0, 1, ..., num_states-1)
        transition_matrix:  [num_states x num_states] matrix, where entry (i, j) is the probability
                            of transitioning from state i to state j
        emission_matrix:    [num_states x num_symbols] matrix, where entry (i, k) is the probability
                            of emitting observation k from state i
        pi:                 Initial state distribution summing to 1, [num_states, ]
        observations:       Sequence of observed symbols (indices from `symbols_set`)

    Returns:
        np.ndarray: Most probable sequence of hidden states
    """

    L = len(observations)
    num_states = len(states_set)

    # Viterbi table (log probabilities to avoid underflow)
    viterbi_log = np.full((L, num_states), -np.inf)      # Stores log probas of being in state s at time t
    backpointer = np.zeros((L, num_states), dtype=int)   # Stores state k from which we arrived to s

    # 1. Initialization: v_k(1) = π_k * e_k(ε_1)
    viterbi_log[0] = np.log(pi) + np.log(emission_matrix[:, observations[0]])

    # 2. Recursion step:  v_k(t) = e_k(ε_{t}) * max_l [ v_l(t-1) * m_{lk} ]
    for t in tqdm(range(1, L), colour='GREEN', desc='Recursion...', disable=tqdm_disable):
        log_probs = viterbi_log[t-1] + np.log(transition_matrix) + np.log(emission_matrix[:, observations[t]])

        # Store the most probable previous state
        backpointer[t] = np.argmax(log_probs, axis=1)

        # Store the best log-probability of this prev state
        viterbi_log[t] = np.max(log_probs, axis=1)

    # 3. Backtracking
    best_path = np.zeros(L, dtype=int)

    # Most probable last state
    best_path[-1] = np.argmax(viterbi_log[-1])

    # For each step we know, from which state we arrived here (most probable)
    for t in range(L-2, -1, -1):
        best_path[t] = backpointer[t+1, best_path[t+1]]

    return best_path


def evaluate(seq_gt: np.ndarray, seq_pred: np.ndarray):
    return (seq_gt != seq_pred).mean()


def bootstrap_viterbi(
    seq: np.ndarray,
    states_gt: np.ndarray,
    mean_length: int,
    hmm_params: dict,
    num_iter: int = 1000,
    ci_level: float = 0.95,
    seed: int = None,
) -> Tuple[dict, np.ndarray]:
    """
    Perform bootstrap analysis of Viterbi algorithm error rate.
    
    Parameters:
        seq:            Observation sequence (nucleotide indices)
        states_gt:      True hidden states (ground truth)
        mean_length:    Average fragment length for bootstrap sampling
        num_iter:       Number of bootstrap iterations (default: 1000)
        ci_level:       Confidence interval level (default: 0.95)
        seed:           Random seed for reproducibility
        
    Returns:
        stats: Dictionary with key statistics:
            - 'mean': mean error rate
            - 'median': median error rate
            - 'std': standard deviation
            - 'ci_low': lower CI bound
            - 'ci_high': upper CI bound
        errors: Array of all error rates (size: num_iter)
    """

    if seed is not None:
        np.random.seed(seed)
    
    errors = []
    rng = np.random.default_rng(seed)
    
    for _ in tqdm(range(num_iter), colour="CYAN", desc="Bootstrap"):
        # Генерация случайного фрагмента
        fragment_length = min(
            max(1, int(rng.exponential(mean_length))),
            len(seq)
        )
        start_pos = rng.integers(0, len(seq) - fragment_length + 1)
        fragment_seq = seq[start_pos:start_pos + fragment_length]
        fragment_gt = states_gt[start_pos:start_pos + fragment_length]
        
        # Предсказание состояний
        best_path = viterbi(
            observations=fragment_seq,
            states_set=hmm_params["states_set"],
            transition_matrix=hmm_params["transition_matrix"],
            pi=hmm_params["pi"],
            emission_matrix=hmm_params["emission_matrix"],
            tqdm_disable=True
        )
        
        # Расчет ошибки
        error_rate = evaluate(seq_gt=fragment_gt, seq_pred=best_path)
        errors.append(error_rate)
    
    errors = np.array(errors)
    
    # Расчет статистик
    mean_err = np.mean(errors)
    median_err = np.median(errors)
    std_err = np.std(errors)
    
    # Доверительный интервал (percentile bootstrap)
    alpha = (1 - ci_level) / 2
    ci_low, ci_high = np.percentile(errors, [100 * alpha, 100 * (1 - alpha)])
    
    stats = {
        'mean': mean_err,
        'median': median_err,
        'std': std_err,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'ci_level': ci_level
    }
    
    return stats, errors

def write_statistics_summary(output_path: Path, 
                           error_stats: dict, 
                           chimera_error: float) -> None:
    """Write error statistics to summary file and log results"""
    logger = setup_logger(__name__, level=logging.INFO)

    # Prepare header
    summary_header = "\n\n=======================\nError Rate Statistics:\n======================="
    
    # Write to file and log
    with open(output_path, 'a') as f:
        # Chimera error
        chimera_msg = f"Chimera Error Rate: {chimera_error:.4f}"
        f.write(f"{summary_header}\n{chimera_msg}\n\n")
        logger.info(summary_header)
        logger.info(chimera_msg)

        # Sequence statistics
        for seq_num, stats in error_stats.items():
            stats_msg = (
                f"Sequence {seq_num} Bootstrap Statistics:\n"
                f"  Mean Error: {stats['mean']:.4f}\n"
                f"  Median Error: {stats['median']:.4f}\n"
                f"  Standard Deviation: {stats['std']:.4f}\n"
                f"  {stats['ci_level']*100:.0f}% Confidence Interval: "
                f"[{stats['ci_low']:.4f}, {stats['ci_high']:.4f}]"
            )
            
            f.write(f"{stats_msg}\n\n")
            logger.info(f"Sequence {seq_num} stats:\n{stats_msg}")
