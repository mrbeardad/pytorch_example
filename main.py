import os
import logging
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import datasets, transforms
import optuna
import rich
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn


def get_data_loader(batch_size: int, train: bool):
    if train:
        train_set = DATASET(
            root="data", train=True, download=True, transform=transforms.ToTensor()
        )
        train_set, valid_set = random_split(train_set, [0.8, 0.2])
        train_loader = DataLoader(
            train_set,
            batch_size,
            shuffle=True,
            num_workers=1,
            persistent_workers=True,
            pin_memory=DEVICE != "cpu",
            pin_memory_device=DEVICE if DEVICE != "cpu" else "",
        )
        valid_loader = DataLoader(
            valid_set,
            batch_size,
            shuffle=True,
            num_workers=1,
            persistent_workers=True,
            pin_memory=DEVICE != "cpu",
            pin_memory_device=DEVICE if DEVICE != "cpu" else "",
        )
        return train_loader, valid_loader

    return DataLoader(
        DATASET(root="data", train=False, download=True, transform=transforms.ToTensor()),
        batch_size,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
        pin_memory=DEVICE != "cpu",
        pin_memory_device=DEVICE if DEVICE != "cpu" else "",
    )


def objective(trial: optuna.Trial):
    batch_size = trial.suggest_categorical("batch_size", [2**x for x in range(2, 10)])
    train_loader, valid_loader = get_data_loader(batch_size, train=True)

    model = define_model(trial).to(DEVICE)
    criterion = CRITERION
    optimizer = OPTIMIZER(
        model.parameters(),
        lr=trial.suggest_float("lr", 1e-5, 1e-1, log=True),
        amsgrad=True,
        fused=True,
    )

    PROGRESS.update(
        PROGRESS_TASK,
        description="[red]Hyperparams Trialing...",
        total=TRIAL_TIMES
        * TRIAL_EPOCHS
        * (TRIAL_TRAIN_EXAMPLES_PER_EPOCH + TRIAL_VALID_EXAMPLES_PER_EPOCH)
        + 1,
    )
    for epoch in range(TRIAL_EPOCHS):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            if batch_idx * batch_size >= TRIAL_TRAIN_EXAMPLES_PER_EPOCH:
                break
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            PROGRESS.update(PROGRESS_TASK, advance=batch_size)

        model.eval()
        correct_count = 0
        total_count = 0
        with torch.inference_mode():
            for batch_idx, (x, y) in enumerate(valid_loader):
                if batch_idx * batch_size >= TRIAL_VALID_EXAMPLES_PER_EPOCH:
                    break
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                y_hat = model(x)
                correct_count += float(torch.sum(y_hat.argmax(1) == y))
                total_count += batch_size
                PROGRESS.update(PROGRESS_TASK, advance=batch_size)

        accuracy = correct_count / total_count
        trial.report(accuracy, epoch)
        if trial.should_prune():
            PROGRESS.update(
                PROGRESS_TASK,
                advance=(TRIAL_EPOCHS - epoch + 1)
                * (TRIAL_TRAIN_EXAMPLES_PER_EPOCH + TRIAL_VALID_EXAMPLES_PER_EPOCH),
            )
            raise optuna.exceptions.TrialPruned()

    return accuracy


def train_model(
    train_loader: DataLoader,
    valid_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    early_stopping_criterion: Literal["both", "loss", "accuracy"] = "both",
    load_checkpoint=False,
):
    checkpoint = os.path.join(WRITER.get_logdir(), "checkpoint.pt")

    min_loss = float("inf")
    max_accuracy = 0.0
    no_descent_count = 0

    epoch = 0
    if load_checkpoint:
        state_dicts = torch.load(checkpoint)
        epoch = state_dicts["epoch"]
        model.load_state_dict(state_dicts["model"])
        optimizer.load_state_dict(state_dicts["optimizer"])
        rich.print("[yellow]Continue training from last checkpoint")

    PROGRESS.update(
        PROGRESS_TASK,
        description="[red]Trialing...",
        total=len(train_loader) + len(valid_loader),
        completed=0,
    )
    # keep training until trigger early stopping
    while True:
        epoch += 1
        rich.print(f"[Epoch: {epoch}]")

        train_loss_sum = 0.0
        valid_loss_sum = 0.0
        valid_correct_count = 0

        # train
        model.train()
        PROGRESS.update(PROGRESS_TASK, description="[red]  Training...")
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            train_loss_sum += loss.item()
            loss.backward()
            optimizer.step()
            PROGRESS.update(PROGRESS_TASK, advance=1)

        # validation
        model.eval()
        PROGRESS.update(PROGRESS_TASK, description="[yellow]Validating...")
        with torch.inference_mode():
            for x, y in valid_loader:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                y_hat = model(x)
                valid_correct_count += float(torch.sum(y_hat.argmax(dim=1) == y))
                loss = criterion(y_hat, y)
                valid_loss_sum += loss.item()
                PROGRESS.update(PROGRESS_TASK, advance=1)

        # epoch information
        avg_train_loss = train_loss_sum / len(train_loader)
        avg_valid_loss = valid_loss_sum / len(valid_loader)
        accuracy = valid_correct_count / len(valid_loader.dataset)
        WRITER.add_scalar("Loss/train", avg_train_loss, epoch)
        WRITER.add_scalar("Loss/valid", avg_valid_loss, epoch)
        WRITER.add_scalar("Accuracy/valid", accuracy, epoch)
        info = (
            f"\tTraining Loss: {avg_train_loss:.4f}, "
            + f"Validation Loss: {avg_valid_loss:.4f}, "
            + f"Validation Accuracy: {accuracy:.2%} "
        )

        # detect early stopping
        if avg_valid_loss <= min_loss and early_stopping_criterion in ("both", "loss"):
            rich.print(
                f"{info} -- Validation Loss decreased ({min_loss:.4f} -> {avg_valid_loss:.4f})"
            )
            min_loss = avg_valid_loss
            no_descent_count = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                checkpoint,
            )
        elif accuracy >= max_accuracy and early_stopping_criterion in ("both", "accuracy"):
            rich.print(
                f"{info} -- Validation Accuracy improved ({max_accuracy:.2%} -> {accuracy:.2%})"
            )
            max_accuracy = accuracy
            no_descent_count = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                checkpoint,
            )
        else:
            rich.print(info)
            no_descent_count += 1
            if no_descent_count > PATIENCE_EPOCHS:
                PROGRESS.update(PROGRESS_TASK, description="[green]Done!", advance=1)
                break

        PROGRESS.update(PROGRESS_TASK, completed=0)

    states = torch.load(checkpoint)
    model.load_state_dict(states["model"])
    optimizer.load_state_dict(states["optimizer"])


def test_model(
    test_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
):
    loss_sum = 0.0
    correct_count = 0
    total_count = 0

    # task = progress.add_task("[yellow]Testing...", total=len(test_loader))
    model.eval()
    with torch.inference_mode():
        for x, y in test_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            y_hat = model(x)
            correct_count += float(torch.sum(y_hat.argmax(1) == y))
            total_count += x.shape[0]
            loss = criterion(y_hat, y)
            loss_sum += loss.item()
            # progress.update(task, advance=1)
    # progress.update(task, description="[green]Done!")

    avg_loss = loss_sum / len(test_loader)
    accuracy = correct_count / total_count
    rich.print("[Test Error]")
    rich.print(f"\tLoss: {avg_loss:.4f} Accuracy: {accuracy:.2%}")


def define_model(trial: optuna.Trial):
    layers = []
    layers.append(nn.Flatten())

    in_features = 28 * 28
    for i in range(trial.suggest_int("hidden_layers", 1, 3)):
        out_features = trial.suggest_int(f"layer{i+1}_units", 10, 512)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.BatchNorm1d(out_features))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(trial.suggest_float(f"layer{i+1}_dropout", 0.2, 0.5)))
        in_features = out_features
    layers.append(nn.Linear(in_features, 10))

    return nn.Sequential(*layers)


def main():
    optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    optuna.logging.disable_default_handler()
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=TRIAL_TIMES, n_jobs=-1)
    best_trial = study.best_trial
    rich.print("Best Hyperparams:")
    rich.print(best_trial.params)

    train_loader, valid_loader = get_data_loader(best_trial.params["batch_size"], train=True)
    model = define_model(best_trial)
    model = (nn.DataParallel(model) if torch.cuda.device_count() > 1 else model).to(DEVICE)
    optimizer = OPTIMIZER(
        model.parameters(),
        lr=best_trial.suggest_float("lr", 1e-5, 1e-1, log=True),
    )
    train_model(train_loader, valid_loader, model, CRITERION, optimizer)


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TRIAL_TIMES = 200
    TRIAL_EPOCHS = 10
    TRIAL_TRAIN_EXAMPLES_PER_EPOCH = 2**12
    TRIAL_VALID_EXAMPLES_PER_EPOCH = 2**9
    PATIENCE_EPOCHS = 20

    DATASET = datasets.MNIST
    CRITERION = nn.CrossEntropyLoss().to(DEVICE)
    OPTIMIZER = torch.optim.AdamW

    WRITER = SummaryWriter(comment=f"-{DATASET.__name__}")

    logging.basicConfig(
        level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
    )
    logger = logging.getLogger("rich")

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as PROGRESS:
        PROGRESS_TASK = PROGRESS.add_task("")
        main()

    WRITER.close()
