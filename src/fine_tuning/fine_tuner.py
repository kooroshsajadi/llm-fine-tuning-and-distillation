from abc import ABC, abstractmethod
from datasets import Dataset
import logging

class FineTuner(ABC):
    """Abstract base class for fine-tuning language models."""

    def __init__(self, logger: logging.Logger):
        """
        Initialize FineTuner with a logger.

        Args:
            logger (logging.Logger): Logger instance for logging.
        """
        self.logger = logger

    @abstractmethod
    def prepare_dataset(self, data_path: str) -> Dataset:
        """
        Prepare dataset for fine-tuning.

        Args:
            data_path (str): Path to the training data file.

        Returns:
            Dataset: Tokenized dataset ready for training.
        """
        pass

    @abstractmethod
    def train(
        self,
        dataset: Dataset,
        output_dir: str,
        per_device_train_batch_size: int,
        num_train_epochs: int,
        learning_rate: float,
        logging_steps: int,
        save_strategy: str
    ) -> None:
        """
        Train the model.

        Args:
            dataset (Dataset): Training dataset.
            output_dir (str): Directory to save model and logs.
            per_device_train_batch_size (int): Batch size per device.
            num_train_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate.
            logging_steps (int): Log every N steps.
            save_strategy (str): Save strategy (e.g., "epoch").
        """
        pass