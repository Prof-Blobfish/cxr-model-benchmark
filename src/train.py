import gc
import torch
from sklearn.metrics import average_precision_score

from train_eval import evaluate, setup_training, train_model
from utils import get_model_path
import config

def run_training_pipeline(
    model_name,
    model_builder,
    train_loader,
    val_loader,
    test_loader,
    device,
    live_plot=False,
):
    print(f"\n=== Training {model_name} ===")
    model = model_builder().to(device)

    criterion, optimizer = setup_training(model)
    
    save_name = f"best_{model_name.lower().replace('-', '_')}"

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        save_name=save_name,
        live_plot=live_plot,
        live_plot_model_name=model_name,
    )

    save_path = get_model_path(save_name)
    model.load_state_dict(torch.load(save_path, map_location=device))

    test_loss, test_acc, test_precision, test_recall, test_f1, test_labels, _, test_probs = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )
    test_auprc = average_precision_score(test_labels, test_probs)

    row = {
        "model": model_name,
        "epochs": len(history["train_loss"]),
        "batch_size": config.BATCH_SIZE,
        "image_size": config.IMAGE_SIZE,
        "test_loss": test_loss,
        "accuracy": test_acc,
        "precision": test_precision,
        "recall": test_recall,
        "f1": test_f1,
        "auprc": test_auprc,
    }

    del model
    gc.collect()

    return row, history