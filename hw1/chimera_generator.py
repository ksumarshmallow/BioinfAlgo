import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

@dataclass
class ChimeraGenerator:
    """
    A class for generation of a chimeric DNA sequence from two input sequences.
    """
    seq1: np.ndarray
    seq2: np.ndarray

    def __post_init__(self):
        self.valid_nucs = np.array(["A", "T", "G", "C"])

        if not isinstance(self.seq1, np.ndarray):
            self.seq1 = np.array(list(self.seq1))
        if not isinstance(self.seq2, np.ndarray):
            self.seq2 = np.array(list(self.seq2))

        self.freqs1 = self._nuc_freqs(self.seq1)
        self.freqs2 = self._nuc_freqs(self.seq2)

        self._fragments_info = []

    def _nuc_freqs(self, seq) -> dict:
        """Calculate nucleotide frequencies"""
        total = len(seq)
        return {nuc.item(): np.sum(seq == nuc).item() / total for nuc in self.valid_nucs}

    def get_nuc_freqs(self, seq) -> dict:
        """Public method to get nucleotide frequencies"""
        return self._nuc_freqs(seq)

    def _sample_fragment_length(self, mean_fragment_length: int) -> int:
        """Sample fragment length ~ Exp(mean_length)"""
        return max(1, int(np.random.exponential(mean_fragment_length)))

    def _get_valid_fragment(self, genome, length) -> np.ndarray:
        """Get a random fragment of specified length"""
        if len(genome) < length:
            return genome
        start = np.random.randint(0, len(genome) - length + 1)
        return genome[start:start + length]

    def generate(self, 
                 mean_fragment_length: int = 300, 
                 max_seqlen: int = 10_000, 
                 seed: int=None,
                 output_folder: Optional[str] = None,
                 output_name: str = "chimera",
                 save_format: str = 'fasta'  # options: 'fasta', 'txt'
                 ) -> tuple[str, str]:
        """Generate chimeric DNA sequence and source mask"""
        
        if seed is not None:
            np.random.seed(seed)

        chimera, sources = [], []
        total_len = 0
        self._fragments_info = []
        genomes = [self.seq1, self.seq2]

        while total_len < max_seqlen:
            source_id = np.random.randint(2)
            genome = genomes[source_id]

            fragment_length = self._sample_fragment_length(mean_fragment_length)
            fragment = self._get_valid_fragment(genome, fragment_length)

            frag_len = len(fragment)
            chimera.append(fragment)
            sources.append(str(source_id + 1) * frag_len)       # source - 1 or 2
            self._fragments_info.append((source_id, frag_len))

            total_len += frag_len

        final_seq = np.concatenate(chimera)[:max_seqlen]
        final_src = "".join(sources)[:max_seqlen]
        chimera_str = "".join(final_seq)
        self._last_chimera = final_seq

        if output_folder:
            self.save(sequence=chimera_str, 
                      states=final_src, 
                      path=output_folder, 
                      filename=output_name, 
                      format=save_format)

        return chimera_str, final_src
    
    def save(self, 
             sequence: str,
             states: str, 
             path: str,
             filename: str,
             format: str = 'fasta'):
        """Save the generated chimera sequence and states"""
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'fasta':
            record = SeqRecord(Seq(sequence), id="chimera", description="Chimeric DNA sequence")
            SeqIO.write(record, path / f"{filename}.fasta", "fasta")
        elif format == 'txt':
            with open(path / f"{filename}.txt", "w") as f:
                f.write(sequence)
        else:
            raise ValueError("Unsupported format. Use 'fasta' or 'txt'.")

        # Save the mask as plain text
        with open(path / f"{filename}_states.txt", "w") as f:
            f.write(states)

    def summary(self):
        """Print statistics on chimeric sequence generation"""
        print("Original frequencies:")
        print("  seq1:", self.freqs1)
        print("  seq2:", self.freqs2)

        if not hasattr(self, "_last_chimera"):
            print("No chimera generated yet.")
            return

        chimera_freqs = self._nuc_freqs(np.array(list(self._last_chimera)))
        print("\nChimera frequencies:")
        print(" ", chimera_freqs)

        frag_lengths = [l for _, l in self._fragments_info]
        src1_total = sum(l for src, l in self._fragments_info if src == 0)
        src2_total = sum(l for src, l in self._fragments_info if src == 1)
        total = src1_total + src2_total
        mean_frag_len = np.mean(frag_lengths)

        print("\nFragment source stats:")
        print(f"  From seq1: {src1_total} nt ({src1_total / total:.2%})")
        print(f"  From seq2: {src2_total} nt ({src2_total / total:.2%})")
        print(f"  Mean fragment length: {mean_frag_len:.2f} nt")
        print(f"  Total fragments: {len(self._fragments_info)}")