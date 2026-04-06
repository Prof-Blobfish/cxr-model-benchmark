from dataclasses import dataclass, field
from typing import List, Optional

# Access examples
# output = experiment_outputs["DenseNet121"]
# print(experiment_outputs["DenseNet121"].metrics.auprc)

@dataclass
class Metrics:
    model: str
    epochs: int
    batch_size: int
    image_size: int
    test_loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    auprc: float

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
    lr: List[float] = field(default_factory=list)
    lr_backbone: List[float] = field(default_factory=list)
    lr_head: List[float] = field(default_factory=list)
    backbone_frozen: List[bool] = field(default_factory=list)
    vram_peak_pct: List[float] = field(default_factory=list)
    best_epoch: Optional[int] = None

@dataclass
class ModelOutput:
    metrics: Metrics
    history: History

