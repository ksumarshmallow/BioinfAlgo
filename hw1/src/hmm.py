import pickle
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from tabulate import tabulate
from dataclasses import dataclass, field

import logging
from utils.logging import setup_logger


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
        self._logger = setup_logger(__name__, level=logging.INFO)

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
        self._logger.info(f'Stationary distribution: {self.pi}')

    def _log_matrix(self, name: str, matrix: np.ndarray, row_labels: np.ndarray, col_labels: np.ndarray):
        """Logging matrix as a table"""
        table = tabulate(matrix, headers=col_labels, tablefmt="grid")
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

class HMMSerializer:
    @staticmethod
    def save(hmm: HMM, path: Path):
        """Save HMM parameters to file"""
        params = {
            'states_set': hmm.states_set,
            'observations_set': hmm.observations_set,
            'transition_matrix': hmm.transition_matrix,
            'emission_matrix': hmm.emission_matrix,
            'pi': hmm.pi,
            'mappings': {
                'obs2idx': hmm.obs2idx,
                'idx2obs': hmm.idx2obs,
                'state2idx': hmm.state2idx,
                'idx2state': hmm.idx2state
            }
        }
        with open(path, 'wb') as f:
            pickle.dump(params, f)

    @staticmethod
    def load(path: Path) -> Dict[str, Any]:
        """Load HMM parameters from file"""
        with open(path, 'rb') as f:
            return pickle.load(f)

