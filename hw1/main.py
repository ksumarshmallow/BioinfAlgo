import sys
import logging
import numpy as np

from sequence_processing import SequenceProcessor
from chimera_generator import ChimeraGenerator
from hmm import HMM, viterbi

from utils.config_parser import ConfigParser
from utils.cli_argument_parser import HW1ParamsParser
from utils.logging import setup_logger

def main(seq1_path: str, seq2_path: str, config: dict, logger: logging.Logger):
    general_params = config_path["GeneralParams"]
    chimera_generator_params = config_path["ChimeraGenerator"]

    valid_nucleotides = general_params['valid_nucleotides']
    logger.info(f'Valid nucleotides: {valid_nucleotides}')

    processor = SequenceProcessor(valid_nucl=valid_nucleotides)

    logger.info(f'Start process sequences on path: {seq1_path} and {seq2_path}')
    seq1 = processor.process(sequence_path=seq1_path)
    seq2 = processor.process(sequence_path=seq2_path)

    chimera_generator = ChimeraGenerator(seq1=seq1, seq2=seq2, valid_nucs=np.array(valid_nucleotides))
    logger.info(f'Start generating chimera sequence \nParams: {chimera_generator_params}')
    chimera_seq, chimera_states = chimera_generator.generate(**chimera_generator_params)

    chimera_generator.summary()

    # always for 2 sequences
    logger.info('Initialize HMM model')
    states_set = np.array([1, 2])
    observations_set = np.array(valid_nucleotides)

    hmm = HMM(states_set=states_set, 
          observations_set=observations_set, 
          freqs_list=[chimera_generator.freqs1, chimera_generator.freqs2],
          mean_steps=chimera_generator_params['mean_fragment_length']
          )

    chimera_seq_coded = np.array([hmm.obs2idx[nucl] for nucl in chimera_seq])

    best_path = viterbi(observations=chimera_seq_coded,
                        states_set=hmm.states_set_coded,
                        transition_matrix=hmm.transition_matrix,
                        pi=hmm.pi,
                        emission_matrix=hmm.emission_matrix)

    best_path_decoded = np.array([hmm.idx2state[i.item()] for i in best_path])
    chimera_states_numpy = np.array([int(i) for i in chimera_states])

    error_rate_chimera = (best_path_decoded != chimera_states_numpy).mean()
    logger.info(f'Error rate in Viterbi algo for Chimera prediction: {error_rate_chimera}')


if __name__ == "__main__":
    # clear logging handlers
    root_logger = logging.root
    root_logger.handlers.clear()
    logging.basicConfig(
        stream=sys.stdout, level=logging.INFO,
        format='[%(asctime)s %(levelname)s] %(name)s: %(message)s'
    )
    logger = logging.getLogger(__name__)

    # CLI params
    params = HW1ParamsParser.parse()

    seq1_path = params["seq1"]
    seq2_path = params["seq2"]
    config_path = params["config"]

    config = ConfigParser.parse(config_path)

    main(seq1_path, seq2_path, config, logger)
