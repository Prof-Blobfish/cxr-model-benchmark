from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
from IPython.display import Markdown, clear_output, display
import ipywidgets as widgets

import config
from train import run_training_pipeline
from train_eval import print_run_configuration, setup_training


def print_planned_run_configuration(
    model_builder: Callable[[], torch.nn.Module],
    device: torch.device,
    model_name: str = "ShuffleNetV2",
    resume_from_checkpoint: bool = False,
) -> None:
    """Print the resolved run config with overrides one-per-line."""
    model = model_builder().to(device)
    _, _, _, training_control = setup_training(model, model_name=model_name)
    print_run_configuration(
        model_name=model_name,
        training_control=training_control,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    
    # Print any model-specific tuning overrides in a clean format
    overrides = config.TUNING_OVERRIDES.get(model_name, {})
    if overrides:
        print(f"\n=== {model_name} Tuning Overrides ===")
        for knob, value in sorted(overrides.items()):
            print(f"  {knob}: {value}")
    
    del model


def run_full_training(
    seed: int,
    run_results: List[Dict[str, Any]],
    baseline_auprc: float,
    model_builder: Callable[[], torch.nn.Module],
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    model_name: str = "ShuffleNetV2",
    live_plot: bool = True,
    resume_from_checkpoint: bool = False,
    epoch_log_path: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[int], Optional[float]]:
    """Run one seed and append structured result data to run_results."""
    metrics, history = run_training_pipeline(
        model_name=model_name,
        model_builder=model_builder,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        live_plot=live_plot,
        resume_from_checkpoint=resume_from_checkpoint,
        show_run_configuration=False,
        epoch_log_path=epoch_log_path,
        epoch_log_prefix=f"Seed {seed}",
    )

    best_epoch = history.get("best_epoch")
    best_val_auprc: Optional[float] = None
    if best_epoch is not None and "val_auprc" in history and history["val_auprc"]:
        best_val_auprc = float(history["val_auprc"][best_epoch - 1])

    print(
        f"Run Seed = {seed} complete: best_epoch={best_epoch}, "
        f"best_val_auprc={best_val_auprc if best_val_auprc is not None else 'N/A'}"
    )

    run_results.append(
        {
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_auprc": best_val_auprc,
            "delta": best_val_auprc - baseline_auprc if best_val_auprc is not None else None,
            "full_metrics": metrics,
            "full_history": history,
        }
    )

    return metrics, history, best_epoch, best_val_auprc


def print_run_metrics(
    best_epoch: Optional[int],
    best_val_auprc: Optional[float],
    metrics: Dict[str, Any],
    history: Dict[str, Any],
    baseline_auprc: float,
) -> None:
    """Print concise post-run metrics for one seed."""
    print("\n=== Run Snapshot ===")
    print(f"Best epoch: {best_epoch if best_epoch is not None else 'N/A'}")
    if best_val_auprc is not None:
        print(f"Best val AUPRC (Delta baseline): {best_val_auprc:.4f} ({best_val_auprc - baseline_auprc:+.4f})")
    else:
        print("Best val AUPRC: N/A")

    test_auprc = metrics.get("auprc")
    if test_auprc is not None:
        print(f"Test AUPRC: {float(test_auprc):.4f}")
    else:
        print("Test AUPRC: N/A")

    print("\n=== Run Summary ===")
    print(f"Model: {metrics.get('model', 'ShuffleNetV2')}")
    print(f"Epochs: {metrics.get('epochs', len(history.get('train_loss', [])))}")

    for key in ["test_loss", "accuracy", "precision", "recall", "f1", "auprc"]:
        value = metrics.get(key)
        if value is None:
            print(f"{key}: N/A")
        else:
            print(f"{key}: {float(value):.4f}")


def render_cumulative_snapshots(run_results: List[Dict[str, Any]]) -> None:
    """Render a compact table of cumulative seed results."""
    if not run_results:
        print("No completed runs yet.")
        return

    summary_df = pd.DataFrame(
        [
            {
                "Seed": r["seed"],
                "Best Epoch": r["best_epoch"],
                "Best Val AUPRC": f"{r['best_val_auprc']:.4f}" if r["best_val_auprc"] is not None else "N/A",
                "Delta": f"{r['delta']:+.4f}" if r["delta"] is not None else "N/A",
            }
            for r in run_results
        ]
    )

    print("=== Cumulative Run Snapshots ===")
    print(summary_df.to_string(index=False))


def run_seed_experiment(
    run_results: List[Dict[str, Any]],
    baseline_auprc: float,
    model_builder: Callable[[], torch.nn.Module],
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    model_name: str = "ShuffleNetV2",
    seeds: Iterable[int] = (16, 32, 64),
    live_plot: bool = True,
    reset_results: bool = False,
    epoch_log_file: str = "../outputs/experiment_outputs/shufflenetv2_epoch_metrics.log",
    reset_epoch_log: bool = True,
) -> None:
    """Run multi-seed training with per-seed panels and cumulative snapshot updates."""
    if reset_results:
        run_results.clear()

    log_path = Path(epoch_log_file).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if reset_epoch_log:
        log_path.write_text("", encoding="utf-8")
        print(f"Epoch log reset: {log_path}")
    else:
        print(f"Appending epoch logs to: {log_path}")

    display(Markdown("### Seed Training Outputs"))
    cumulative_out = widgets.Output()
    display(Markdown("### Stacked Snapshots"))
    display(cumulative_out)

    for idx, seed in enumerate(seeds, start=1):
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"=== Run {idx} | Seed {seed} ===\\n")
            handle.flush()

        seed_out = widgets.Output()
        display(Markdown(f"#### Run {idx} | Seed {seed}"))
        display(seed_out)

        with seed_out:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            metrics, history, best_epoch, best_val_auprc = run_full_training(
                seed=seed,
                run_results=run_results,
                baseline_auprc=baseline_auprc,
                model_builder=model_builder,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                model_name=model_name,
                live_plot=live_plot,
                resume_from_checkpoint=False,
                epoch_log_path=str(log_path),
            )
            print_run_metrics(
                best_epoch=best_epoch,
                best_val_auprc=best_val_auprc,
                metrics=metrics,
                history=history,
                baseline_auprc=baseline_auprc,
            )

        with cumulative_out:
            clear_output(wait=True)
            render_cumulative_snapshots(run_results)


def cleanup_training_artifacts() -> None:
    """Free common transient training objects and clear CUDA cache when available."""
    import gc
    import time

    for name in [
        "model",
        "optimizer",
        "scheduler",
        "criterion",
        "batch",
        "images",
        "labels",
    ]:
        if name in globals():
            del globals()[name]

    for _ in range(3):
        gc.collect()
        time.sleep(0.1)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        print("CUDA cache cleared.")
    else:
        print("CUDA not available; CPU cleanup complete.")

    print("Cleanup complete. Loaders preserved for next run.")


def cleanup() -> None:
    """Backward-compatible alias for notebook cleanup calls."""
    cleanup_training_artifacts()
