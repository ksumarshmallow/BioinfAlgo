import sys
import logging
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.sequence_processing import SequenceProcessor
from src.chimera_generator import ChimeraGenerator
from src.hmm import HMM
from src.viterbi import viterbi, evaluate, bootstrap_viterbi, write_statistics_summary

from utils.config_parser import ConfigParser
from utils.cli_argument_parser import HW1ParamsParser
from utils.logging import setup_logger


def main(seq1_path: str, seq2_path: str, config: dict, logger: logging.Logger):
    """Main processing pipeline"""
    
    # Configuration setup
    chimera_params = config["ChimeraGenerator"]
    valid_nucleotides = config['valid_nucleotides']
    
    # Output configuration
    output_folder = Path(chimera_params['output_folder'])

    output_folder.mkdir(parents=True, exist_ok=True)

    output_name = chimera_params['output_name']

    summary_path = output_folder / "summary.txt"
    viterbi_state_path = output_folder / f"{output_name}_states_viterbi.txt"
    emission_matrix_path = output_folder / "emission_matrix.npy"

    # Step 1. Process input sequences
    logger.info(f'Processing sequences: {seq1_path} and {seq2_path}')
    processor = SequenceProcessor(valid_nucl=valid_nucleotides)
    seq1, seq2 = [processor.process(p) for p in (seq1_path, seq2_path)]

    # Step 2. Generate chimera sequence
    logger.info('Start generating Chimera sequence')
    chimera_gen = ChimeraGenerator(seq1, seq2, np.array(valid_nucleotides))
    chimera_seq, chimera_states = chimera_gen.generate(**chimera_params)
    chimera_gen.summary(output_folder=output_folder, filename=summary_path.name)

    # Step 3. HMM initialization and processing
    logger.info('Initializing HMM model')
    hmm = HMM(
        states_set=np.array([1, 2]),
        observations_set=np.array(valid_nucleotides),
        freqs_list=[chimera_gen.freqs1, chimera_gen.freqs2],
        mean_steps=config["ChimeraGenerator"]['mean_fragment_length']
    )

    hmm_params = {
        "states_set": hmm.states_set,
        "transition_matrix": hmm.transition_matrix,
        "emission_matrix": hmm.emission_matrix,
        "pi": hmm.pi
    }

    with open(summary_path, 'a') as f:
        f.write("\n\n=======================\n")
        f.write("HMM Parameters:\n")
        f.write("=======================\n")

        for k, v in hmm_params.items():
            f.write(f"{k}:\n{v}\n\n")
    
    np.save(emission_matrix_path, hmm.emission_matrix)
    logger.info(f"Emission matrix saved in path: {emission_matrix_path}")

    # Step 4. Viterbi analysis
    logger.info('Running Viterbi algorithm')
    chimera_coded = np.array([hmm.obs2idx[nucl] for nucl in chimera_seq])
    best_path = viterbi(observations=chimera_coded, **hmm_params)
    best_path_decoded = np.array([hmm.idx2state[i.item()] for i in best_path])

    with open(viterbi_state_path, 'w') as f:
        f.write("".join(best_path_decoded.astype(str)))
    logger.info(f'Predicted states saved to {viterbi_state_path}')

    # Step 5. Error calculation
    chimera_states_np = np.array([int(i) for i in chimera_states])
    error_rate = evaluate(chimera_states_np, best_path_decoded)
    logger.info(f'Viterbi error rate: {error_rate:.4f}')

    # Bootstrap analysis
    seq1_coded = np.array([hmm.obs2idx[i] for i in seq1])
    seq2_coded = np.array([hmm.obs2idx[i] for i in seq2])

    stats_seq1, _ = bootstrap_viterbi(
        seq=seq1_coded,
        hmm_params=hmm_params,
        states_gt=np.zeros(len(seq1_coded)),
        mean_length=chimera_params['mean_fragment_length'], 
        num_iter=1000,
        seed=chimera_params['seed']
    )
    
    stats_seq2, _ = bootstrap_viterbi(
        seq=seq2_coded, 
        hmm_params=hmm_params,
        states_gt=np.ones(len(seq2_coded)),
        mean_length=chimera_params['mean_fragment_length'], 
        num_iter=1000,
        seed=chimera_params['seed']
    )

    # Save statistics
    write_statistics_summary(summary_path, {'1': stats_seq1, '2': stats_seq2}, error_rate)

if __name__ == "__main__":
    logger = setup_logger(__name__, level=logging.INFO)

    params = HW1ParamsParser.parse()
    config = ConfigParser.parse(params["config"])

    main(params["seq1"], params["seq2"], config, logger)
