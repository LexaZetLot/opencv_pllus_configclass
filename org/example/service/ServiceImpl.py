from abc import ABC, abstractmethod


class ServiceImpl(ABC):
    @abstractmethod
    def run(self) -> None:
        pass