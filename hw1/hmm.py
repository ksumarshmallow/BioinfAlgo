import numpy as np
from tqdm import tqdm
from typing import List
from logging import getLogger
from tabulate import tabulate
from dataclasses import dataclass, field


@dataclass
class HMM:
    states_set: np.ndarray
    observations_set: np.ndarray
    freqs_list: List[dict]                              # Список частот для каждого генома
    mean_steps: int = 300                               # Среднее количество шагов
    transition_matrix: np.ndarray = field(init=False)
    emission_matrix: np.ndarray = field(init=False)
    pi: np.ndarray = field(init=False)                  # Начальное распределение состояний

    def __post_init__(self):
        self._logger = getLogger(__name__)

        self.state2idx = {state.item(): i for i, state in enumerate(self.states_set)}
        self.obs2idx = {nucl.item(): i for i, nucl in enumerate(self.observations_set)}
        self.idx2state = {i: state.item() for i, state in enumerate(self.states_set)}
        self.idx2obs = {i: nucl.item() for i, nucl in enumerate(self.observations_set)}

        self.states_set_coded = [self.state2idx[state] for state in self.states_set]
        self.observations_set_coded = [self.obs2idx[nucl] for nucl in self.observations_set]
        
        self._prepare_matrices()

    def _prepare_matrices(self):
        self.transition_matrix = self._get_transition_matrix()
        self.emission_matrix = self._get_emission_matrix(self.freqs_list)
        self.pi = self._get_stationary_distribution(self.transition_matrix)

        self._log_matrix('Transition matrix', self.transition_matrix, self.states_set, self.states_set)
        self._log_matrix('Emission matrix', self.emission_matrix, self.states_set, self.observations_set)
        self._logger.info(f'Stationary distribution: \n{self.pi}')
    
    def _log_matrix(self, name: str, matrix: np.ndarray, row_labels: np.ndarray, col_labels: np.ndarray):
        """Logging matrix as a table"""
        table = tabulate(matrix, headers=col_labels, showindex=row_labels, tablefmt="grid")
        self._logger.info(f"{name}:\n{table}")

    def _get_transition_matrix(self):
        """Transition matrix distribution based on mean step value"""
        lambda_ = 1 / self.mean_steps  # eponentioal distribution
        return np.array([[1 - lambda_, lambda_], [lambda_, 1 - lambda_]])

    def _get_emission_proba(self, freqs):
        """Emission proba calculation according to nucleotide genome frequencies"""
        gc_proba = (freqs['G'] + freqs['C']) / 2
        at_proba = (freqs['A'] + freqs['T']) / 2
        return gc_proba, at_proba

    def _get_emission_matrix(self, freqs_list):
        """Emission matrix generation according to nucleotide genome frequencies"""
        num_states = len(self.states_set)
        emission_matrix = np.zeros((num_states, len(self.observations_set)))

        for idx, freqs in enumerate(freqs_list):
            gc_proba, at_proba = self._get_emission_proba(freqs)

            # Заполнение матрицы для каждого состояния
            emission_matrix[idx, self.obs2idx['G']] = gc_proba
            emission_matrix[idx, self.obs2idx['C']] = gc_proba
            emission_matrix[idx, self.obs2idx['A']] = at_proba
            emission_matrix[idx, self.obs2idx['T']] = at_proba

        return emission_matrix

    def _get_stationary_distribution(self, transition_matrix: np.ndarray):
        eig_values, eig_vectors = np.linalg.eig(transition_matrix.T)
        pi = np.real(eig_vectors[:, np.argmax(np.isclose(eig_values, 1))])
        pi /= pi.sum()
        return pi


def viterbi(states_set: np.ndarray,
            transition_matrix: np.ndarray,
            emission_matrix: np.ndarray,
            pi: np.ndarray,
            observations: np.ndarray):
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
    for t in tqdm(range(1, L), colour='GREEN', desc='Recursion...'):
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