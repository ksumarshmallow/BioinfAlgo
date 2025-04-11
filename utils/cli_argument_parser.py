from argparse import ArgumentParser

class HW1ParamsParser:
    @staticmethod
    def parse() -> dict:
        """
        Get command line arguments
        :return: argparse arguments
        """
        argument_parser = ArgumentParser(description='CLI')
        argument_parser.add_argument(
            '--seq1', required=False, help='Path to file with sequence-1 (.txt or .fasta)'
        )

        argument_parser.add_argument(
            '--seq2', required=False, help='Path to file with sequence-2 (.txt or .fasta)'
        )

        argument_parser.add_argument(
            '--config', required=True, help='Path to config with processing and model params'
        )

        return vars(
            argument_parser.parse_args()
        )

class HW1ViterbiParamsParser:
    @staticmethod
    def parse() -> dict:
        """
        Get command line arguments
        :return: argparse arguments
        """
        argument_parser = ArgumentParser(description='CLI')
        argument_parser.add_argument(
            '--seq', required=False, help='Path to file with sequence (.txt or .fasta)'
        )

        argument_parser.add_argument(
            '--config', required=True, help='Path to config with processing and model params'
        )
        
        argument_parser.add_argument(
            '--output', required=True, help='Output path for predicted states'
        )

        return vars(
            argument_parser.parse_args()
        )