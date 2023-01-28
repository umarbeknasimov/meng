import argparse
import abc

class Runner:
    """instance of a training run"""

    @staticmethod
    @abc.abstractmethod
    def description() -> str:
        """a description of this runner"""
        pass

    # @staticmethod
    # @abc.abstractmethod
    # def add_args(parser: argparse.ArgumentParser) -> None:
    #     """add command line flags for this runner"""
    #     pass

    @staticmethod
    @abc.abstractmethod
    def create_from_args(args: argparse.Namespace) -> 'Runner':
        """create a runner from command line arguments"""

        pass

    @abc.abstractmethod
    def display_output_location(self) -> None:
        "print output location for this job"

        pass

    @abc.abstractmethod
    def run(self) -> None:
        """run the job"""
        pass
    
