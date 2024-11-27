from abc import ABC, abstractmethod


class BaseProcessor(ABC):
    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def statistical_summary(self):
        pass

    @abstractmethod
    def preprocess_data(self):
        pass

    @abstractmethod
    def extract_features(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def evaluate_model(self):
        pass

    @abstractmethod
    def visualize_data(self):
        pass

    @abstractmethod
    def generate_report(self):
        pass
