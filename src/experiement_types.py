from dataclasses import dataclass
from typing import List, Optional

# Access examples
# output = experiment_outputs["DenseNet121"]
# print(experiment_outputs["DenseNet121"].metrics.auprc)

@dataclass
class Metrics:
    model: str
    test_loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    auprc: float
    epochs: int

@dataclass
class History:
    train_loss: List[float]
    train_acc: List[float]
    val_loss: List[float]
    val_acc: List[float]
    val_precision: List[float]
    val_recall: List[float]
    val_f1: List[float]
    val_auprc: List[float]
    best_epoch: Optional[int]

@dataclass
class ModelOutput:
    metrics: Metrics
    history: History

