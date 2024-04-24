from abc import ABC, abstractmethod

from tensorboardX import SummaryWriter


class TensorBoardBaseLogger(ABC):
    def __init__(self, writer: SummaryWriter, metric_name: str, metric_category: str):
        self._writer = writer
        self._prefix = f'{metric_category}/{metric_name}'

    def log(self, data, step):
        processed_data = self._process_data(data)
        self._writer.add_scalar(self._prefix, processed_data, step)

    @abstractmethod
    def _process_data(self, data) -> float:
        pass
