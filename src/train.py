import gc
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import average_precision_score

from train_eval import evaluate, setup_training, train_model, print_run_configuration
from utils import (
    get_model_path,
    get_training_checkpoint_path,
    get_experiment_outputs_path,
    load_experiment_outputs,
    save_experiment_outputs,
)
from experiement_types import Metrics, History, ModelOutput
import config


def _make_smoke_loader(loader, smoke_batches: int) -> DataLoader:
    """Build a tiny loader capped to a small number of batches for fast smoke checks."""
    if smoke_batches < 1:
        raise ValueError("smoke_batches must be >= 1")

    batch_size = loader.batch_size if loader.batch_size is not None else 1
    max_items = min(len(loader.dataset), batch_size * smoke_batches)
    subset = Subset(loader.dataset, list(range(max_items)))

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=getattr(loader, "pin_memory", False),
        collate_fn=loader.collate_fn,
        drop_last=False,
    )

def run_training_pipeline(
    model_name,
    model_builder,
    train_loader,
    val_loader,
    test_loader,
    device,
    live_plot=False,
    resume_from_checkpoint=None,
    show_run_configuration=True,
    epoch_log_path=None,
    epoch_log_prefix=None,
):
    print(f"\n=== Training {model_name} ===")
    model = model_builder().to(device)

    criterion, optimizer, scheduler, training_control = setup_training(
        model,
        model_name=model_name,
    )

    if resume_from_checkpoint is None:
        resume_from_checkpoint = config.AUTO_RESUME_TRAINING

    if show_run_configuration:
        print_run_configuration(
            model_name=model_name,
            training_control=training_control,
            resume_from_checkpoint=resume_from_checkpoint,
        )
    
    save_name = f"best_{model_name.lower().replace('-', '_')}"
    checkpoint_path = get_training_checkpoint_path(save_name)

    history = train_model(
        model_name=model_name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        training_control=training_control,
        device=device,
        save_name=save_name,
        checkpoint_path=checkpoint_path,
        resume_from_checkpoint=resume_from_checkpoint,
        patience=int(training_control.get("patience", config.PATIENCE)),
        live_plot=live_plot,
        epoch_log_path=epoch_log_path,
        epoch_log_prefix=epoch_log_prefix,
    )

    save_path = get_model_path(save_name)
    model.load_state_dict(torch.load(save_path, map_location=device))

    test_loss, test_acc, test_precision, test_recall, test_f1, test_labels, _, test_probs = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        split_name="Test",
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


def store_model_output(model_name, metrics, history, output_path=None):
    """Persist a model's metrics/history into the experiment outputs store."""
    resolved_output_path = output_path if output_path else get_experiment_outputs_path()
    experiment_outputs = load_experiment_outputs(resolved_output_path)

    experiment_outputs[model_name] = ModelOutput(
        metrics=Metrics(**metrics),
        history=History(**history),
    )
    save_experiment_outputs(experiment_outputs, resolved_output_path)
    return resolved_output_path

def run_smoke_test(
    model_name,
    model_builder,
    train_loader,
    val_loader,
    test_loader,
    device,
    smoke_epochs=1,
    smoke_patience=1,
    smoke_batches=1,
    live_plot=False,
    persist_outputs=False,
    persist_model_name=None,
    output_path=None,
):
    """Run a fast end-to-end sanity check using the normal training pipeline."""
    if smoke_epochs < 1:
        raise ValueError("smoke_epochs must be >= 1")
    if smoke_patience < 1:
        raise ValueError("smoke_patience must be >= 1")
    if smoke_batches < 1:
        raise ValueError("smoke_batches must be >= 1")

    original_epochs = config.NUM_EPOCHS
    original_patience = config.PATIENCE

    print(
        f"\n=== Smoke test: {model_name} "
        f"(epochs={smoke_epochs}, patience={smoke_patience}, batches={smoke_batches}) ==="
    )

    try:
        config.NUM_EPOCHS = smoke_epochs
        config.PATIENCE = smoke_patience

        smoke_train_loader = _make_smoke_loader(train_loader, smoke_batches)
        smoke_val_loader = _make_smoke_loader(val_loader, smoke_batches)
        smoke_test_loader = _make_smoke_loader(test_loader, smoke_batches)

        metrics, history = run_training_pipeline(
            model_name=model_name,
            model_builder=model_builder,
            train_loader=smoke_train_loader,
            val_loader=smoke_val_loader,
            test_loader=smoke_test_loader,
            device=device,
            live_plot=live_plot,
            resume_from_checkpoint=False,
        )

        if persist_outputs:
            save_key = persist_model_name or model_name
            persisted_path = store_model_output(
                model_name=save_key,
                metrics=metrics,
                history=history,
                output_path=output_path,
            )

            reloaded_outputs = load_experiment_outputs(persisted_path)
            reloaded_output = reloaded_outputs.get(save_key)
            if reloaded_output is None:
                raise RuntimeError(
                    f"Smoke test persistence check failed: '{save_key}' not found in {persisted_path}."
                )

            if reloaded_output.metrics.model != metrics["model"]:
                raise RuntimeError(
                    "Smoke test persistence check failed: reloaded metrics do not match saved values."
                )

            if len(reloaded_output.history.train_loss) != len(history["train_loss"]):
                raise RuntimeError(
                    "Smoke test persistence check failed: reloaded history length does not match saved history."
                )

            print(f"Smoke test persistence verified for '{save_key}' at {persisted_path}")

        return metrics, history
    finally:
        config.NUM_EPOCHS = original_epochs
        config.PATIENCE = original_patience