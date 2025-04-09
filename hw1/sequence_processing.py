import numpy as np
from Bio import SeqIO
from pathlib import Path
from dataclasses import dataclass

@dataclass
class SequenceProcessor:
    """
    A class for processing genomic sequences.
    
    This class accepts either:
      - a FASTA file (a file with a header and sequence)
      - or a file that contains only the nucleotide sequence
    """
    
    def __post_init__(self):
        self.VALID_NUCS = set("ATGC")


    def load_sequence(self, seq_path: Path) -> str:
        """
        Loads the sequence from the file located at 'seq_path'.
        It attempts to parse the file as FASTA using BioPython's SeqIO.
        If no record is found or parsing fails, it reads the file as raw text.
        
        Parameters:
            seq_path (Path): The path to the sequence file.
            
        Returns:
            str: The loaded sequence string.
        """
        records = list(SeqIO.parse(seq_path, "fasta"))
        if records:
            return str(records[0].seq)
        return seq_path.read_text()


    def process_fasta(self, fasta_data: str) -> str:
        """
        Processes FASTA-formatted data into a nucleotide sequence.
        
        Steps:
          - Removes header lines (lines starting with '>')
          - Removes spaces and newline characters
          - Converts the sequence to uppercase
          - Filters out any non-ATGC characters using the clean_sequence method
        """

        lines = fasta_data.splitlines()
        seq_lines = [line.strip() for line in lines if not line.startswith(">")]
        return self.clean_sequence("".join(seq_lines).upper())


    def clean_sequence(self, seq: str) -> str:
        """
        Filters the sequence, returning a new string containing only the valid nucleotides (A, T, G, C)
        """
        return "".join(nuc for nuc in seq if nuc in self.VALID_NUCS)


    def process(self, sequence_path: str = None) -> np.ndarray:
        """
        Main processing method that accepts a path to file with sequence ('sequence_path').

        Returns:
            np.ndarray: A NumPy array of individual nucleotide characters.
        """
        if sequence_path is None:
            raise ValueError("Argument 'sequence_path' is required")
        
        seq_path = Path(sequence_path)
        file_content = self.load_sequence(seq_path)

        if file_content.lstrip().startswith(">"):
            sequence = self.process_fasta(file_content)
        else:
            sequence = self.clean_sequence("".join(file_content.split()))
        return np.array(list(sequence))
