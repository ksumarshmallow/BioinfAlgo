import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from src.hmm import HMMSerializer
from src.viterbi import viterbi
from src.sequence_processing import SequenceProcessor

def run_viterbi_pipeline(
    input_seq: str,
    hmm_params_path: Optional[Path] = None,
    custom_params: Optional[Dict] = None,
    output_path: Optional[Path] = None
):
    if hmm_params_path:
        params = HMMSerializer.load(hmm_params_path)
    elif custom_params:
        params = custom_params
    else:
        raise ValueError("Must provide either 'hmm_params_path' or 'custom_params'")
    
    processor = SequenceProcessor(valid_nucl=list(params['observations_set']))
    seq_array = processor.clean_sequence(input_seq)
    seq_coded = np.array([params['mappings']['obs2idx'][nuc] for nuc in seq_array])
    
    best_path = viterbi(
        states_set=np.arange(len(params['states_set'])),
        transition_matrix=params['transition_matrix'],
        emission_matrix=params['emission_matrix'],
        pi=params['pi'],
        observations=seq_coded
    )
    
    decoded_states = [params['mappings']['idx2state'][idx] for idx in best_path]
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(''.join(map(str, decoded_states)))
    
    return decoded_states

if __name__ == "__main__":
    ...