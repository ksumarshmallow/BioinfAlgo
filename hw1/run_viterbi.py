import sys
import logging
import numpy as np
from pathlib import Path
from typing import Optional

from src.viterbi import viterbi
from src.sequence_processing import SequenceProcessor

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.config_parser import ConfigParser
from utils.cli_argument_parser import HW1ViterbiParamsParser
from utils.logging import setup_logger

def main(
    path_seq: str,
    config: dict,
    output_path: Optional[Path]=None,
    logger: logging.Logger=None
):
    
    valid_nucl = config["valid_nucleotides"]
    mean_steps = config["mean_steps"]
    states = [1, 2]

    obs2idx = {nucl: i for i, nucl in enumerate(valid_nucl)}
    idx2obs = {i: nucl for i, nucl in enumerate(valid_nucl)}
    idx2state = {i: state for i, state in enumerate(states)}
    state2idx = {state: i for i, state in enumerate(states)}

    states_coded = [state2idx[i] for i in states]
    obs_coded = [obs2idx[i] for i in valid_nucl]
    
    logger.info(f'Processing sequence: {path_seq}')
    processor = SequenceProcessor(valid_nucl=valid_nucl)
    seq_array = processor.process(path_seq)
    seq_coded = np.array([obs2idx[nuc] for nuc in seq_array])

    logger.info('Load & Calculate HMM params')
    
    emission_matrix = np.load(config["emission_matrix_path"])

    lambda_ = 1 / mean_steps  # exponential distribution
    transition_matrix = np.array([[1 - lambda_, lambda_], [lambda_, 1 - lambda_]])

    pi = [0.5, 0.5]   # uniform distribution

    logger.info(f"Emission matrix:\n{emission_matrix}")
    logger.info(f"Transition matrix:\n{transition_matrix}")
    logger.info(f"Statinary distribution:\n{pi}")


    logger.info('Running Viterbi algorithm')
    best_path = viterbi(
        states_set=states_coded,
        transition_matrix=transition_matrix,
        emission_matrix=emission_matrix,
        pi=pi,
        observations=seq_coded
    )
    
    best_path_decoded = [idx2state[idx] for idx in best_path]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(''.join(map(str, best_path_decoded)))

        logger.info(f'Predicted states saved to {output_path}')

if __name__ == "__main__":
    logger = setup_logger(__name__, level=logging.INFO)

    params = HW1ViterbiParamsParser.parse()
    config = ConfigParser.parse(params["config"])

    main(path_seq=params["seq"], 
         output_path=Path(params["output"]),
         config=config,
         logger=logger)