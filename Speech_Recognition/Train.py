import time
import json
import sys
import matplotlib.pyplot as plt
import mlflow
from mlflow.client import MlflowClient
from tqdm import tqdm
import numpy as np
from functools import partial

import torch
from torch import nn, optim
from torch.optim import Adam
import torch.utils.data as data

from TransformerEncoder import Transformer
from Dataset import DataClass, collate_fn
from BatchSampler import CustomBatchSampler

sys.path.append("/workspace")


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.kaiming_uniform_(m.weight.data)


def get_latest_registered_model(client, experiment_name):
    registered_models = client.search_registered_models()
    if not registered_models:
        print("No registered models found.")
        return None, None, None

    # Sort registered models by latest version creation timestamp
    latest_model = None
    latest_version = None
    latest_timestamp = 0

    for model in registered_models:
        for version in model.latest_versions:
            if (
                version.current_stage == "Production"
                and version.creation_timestamp > latest_timestamp
            ):
                latest_model = model
                latest_version = version
                latest_timestamp = version.creation_timestamp

    # :TODO: Change to use version instead of name
    # Pull the respective validation loss
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    filter_string = f"tags.mlflow.runName = '{latest_model.name}'"
    latest_run = client.search_runs(
        experiment_ids=[experiment_id], filter_string=filter_string
    )
    latest_val_loss = latest_run.to_list()[0].data.metrics["Validation_Loss"]

    if latest_model and latest_version:
        return latest_model, latest_version, latest_val_loss
    else:
        print("No registered model in production stage found.")
        return None, None, None


def load_model_weights_from_run(run_id):
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    return model


# Function to read loss values from a file
def read_loss_file(filepath):
    with open(filepath, "r") as file:
        loss_values = file.read().strip()
    # Parse the JSON-like list
    loss_values = json.loads(loss_values)
    return loss_values


def initialize_model_and_data(client, experiment_name, num_workers):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using {device} for training...")
    torch.manual_seed(1)

    sample_rate = 44100
    train_data_path = "dataset/Train/"
    validation_data_path = "dataset/Validation/"
    path_to_save_runs = "training_runs"
    batch_size = 32
    eval_batch_size = 64
    num_batches = 500

    # enc_voc_size in this case is the number of features of MFCC
    enc_voc_size = 40
    src_pad_idx = 0
    # d_model here is the dimension that positional encoding maps to
    d_model = 256
    max_len = 862
    ffn_hidden = 128
    n_heads = 4
    n_layers = 3
    drop_prob = 0.1

    train_dataset = DataClass(
        data_path=train_data_path,
        sample_rate=sample_rate,
        max_len=max_len,
        valid=False,
    )
    validation_dataset = DataClass(
        data_path=validation_data_path,
        sample_rate=sample_rate,
        max_len=max_len,
        valid=True,
    )

    kwargs = {"num_workers": num_workers, "pin_memory": True} if use_cuda else {}

    print("Loading train")
    train_sampler = CustomBatchSampler(
        train_dataset, batch_size
    )  #  num_batches=num_batches)
    validation_sampler = CustomBatchSampler(validation_dataset, eval_batch_size)

    print("Loading valid")
    # Create a partial function with fixed arguments
    collate_fn_with_args = partial(collate_fn, max_len=max_len, valid=False)

    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn_with_args,
        **kwargs,
    )

    # Create a partial function with fixed arguments
    collate_fn_with_args = partial(collate_fn, max_len=max_len, valid=True)

    validation_loader = data.DataLoader(
        dataset=validation_dataset,
        batch_sampler=validation_sampler,
        collate_fn=collate_fn_with_args,
        **kwargs,
    )

    model = Transformer(
        src_pad_idx=src_pad_idx,
        d_model=d_model,
        enc_voc_size=enc_voc_size,
        max_len=max_len,
        ffn_hidden=ffn_hidden,
        n_head=n_heads,
        n_layers=n_layers,
        drop_prob=drop_prob,
        device=device,
    ).to(device)

    print(f"The model has {count_parameters(model):,} trainable parameters")

    latest_model, latest_version, latest_val_loss = get_latest_registered_model(
        client, experiment_name
    )

    if latest_model and latest_version:
        print(
            f"Loading weights from model '{latest_model.name}' version {latest_version.version}"
        )
        model = load_model_weights_from_run(latest_version.run_id)
    else:
        model.apply(initialize_weights)

    init_lr = 1e-5
    weight_decay = 5e-4
    adam_eps = 5e-9
    warmup = 3
    clip = 1.0

    run_parameters = {
        "sample_rate": sample_rate,
        "batch_size": batch_size,
        "enc_voc_size": enc_voc_size,
        "d_model": d_model,
        "max_len": max_len,
        "ffn_hidden": ffn_hidden,
        "init_lr": init_lr,
    }

    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    optimizer = Adam(
        params=model.parameters(), lr=init_lr, weight_decay=weight_decay, eps=adam_eps
    )

    factor = 0.9
    patience = 10
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, factor=factor, patience=patience
    )

    criterion = nn.BCEWithLogitsLoss()

    return (
        device,
        warmup,
        clip,
        scheduler,
        criterion,
        run_parameters,
        path_to_save_runs,
        train_loader,
        validation_loader,
        model,
        optimizer,
        latest_model,
        latest_version,
        latest_val_loss,
    )


def train(model, iterator, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0

    # Initialize tqdm progress bar
    progress_bar = tqdm(iterator, desc="Training", leave=False)
    acc_outputs = []
    acc_labels = []

    for i, (x, labels) in enumerate(tqdm(progress_bar)):

        optimizer.zero_grad()
        output = model(x.to(device))
        acc_outputs.extend(torch.sigmoid(output).cpu().detach().numpy())
        acc_labels.extend(labels.cpu().detach().numpy())

        loss = criterion(output, labels.float().to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

        # Update tqdm progress bar with current loss
        progress_bar.set_postfix(loss=loss.item())

    predicted_labels = (np.array(acc_outputs) > 0.5).astype(np.float32)  # .float()
    correct_predictions = (np.array(predicted_labels) == np.array(acc_labels)).astype(
        np.float32
    )
    accuracy = correct_predictions.mean().item()

    return epoch_loss / len(iterator), accuracy


def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0

    # Initialize tqdm progress bar
    progress_bar = tqdm(iterator, desc="Validation", leave=False)
    acc_outputs = []
    acc_labels = []

    with torch.no_grad():
        for i, (x, labels) in enumerate(tqdm(progress_bar)):

            output = model(x.to(device))
            acc_outputs.extend(torch.sigmoid(output).cpu().detach().numpy())
            acc_labels.extend(labels.cpu().detach().numpy())

            loss = criterion(output, labels.float().to(device))
            epoch_loss += loss.item()

            # Update tqdm progress bar with current loss
            progress_bar.set_postfix(loss=loss.item())

    predicted_labels = (np.array(acc_outputs) > 0.5).astype(np.float32)
    correct_predictions = (np.array(predicted_labels) == np.array(acc_labels)).astype(
        np.float32
    )
    accuracy = correct_predictions.mean().item()

    return epoch_loss / len(iterator), accuracy


def run_training(
    path_to_save_runs,
    device,
    total_epoch,
    best_loss,
    train_loader,
    validation_loader,
    model,
    warmup,
    clip,
    optimizer,
    scheduler,
    criterion,
    latest_val_loss,
):
    train_losses, validation_losses = [], []
    for step in range(total_epoch):
        start_time = time.time()
        train_loss, train_accuracy = train(
            model, train_loader, optimizer, criterion, clip, device
        )
        valid_loss, valid_accuracy = evaluate(
            model, validation_loader, criterion, device
        )
        end_time = time.time()

        if step > warmup:
            scheduler.step(valid_loss)

        train_losses.append(train_loss)
        validation_losses.append(valid_loss)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss
            print("Saving model...")
            torch.save(
                model.state_dict(), f"{path_to_save_runs}/saved/model_overfit.pt"
            )

        f = open(f"{path_to_save_runs}/result/train_loss.txt", "w")
        f.write(str(train_losses))
        f.close()

        f = open(f"{path_to_save_runs}/result/test_loss.txt", "w")
        f.write(str(validation_losses))
        f.close()

        print(f"Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f}")
        print(f"\tVal Loss: {valid_loss:.3f}")

        # Log desired metrics
        mlflow.log_metrics(
            {
                "Training_Loss": np.round(train_loss, 2),
                "Validation_Loss": np.round(valid_loss, 2),
                "Training_Accuracy": np.round(train_accuracy * 100, 2),
                "Validation_Accuracy": np.round(valid_accuracy * 100, 2),
            }
        )

    # Simplistic approach to update the registered model based only on the last
    # validation loss value
    if valid_loss <= latest_val_loss:
        upgrade_to_production = True
    else:
        upgrade_to_production = False

    # Log the model
    mlflow.pytorch.log_model(model, "model")

    # Read training and validation loss values
    train_loss = read_loss_file(f"{path_to_save_runs}/result/train_loss.txt")
    test_loss = read_loss_file(f"{path_to_save_runs}/result/test_loss.txt")

    # Create an index for the x-axis
    epochs = range(1, len(train_loss) + 1)

    # Plot the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(
        epochs, train_loss, "b-", label="Training Loss"
    )  # blue line for training loss
    plt.plot(
        epochs, test_loss, "r-", label="Validation Loss"
    )  # red line for validation loss

    # Adding labels and title
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Show the plot
    plt.savefig(f"{path_to_save_runs}/result/losses.jpg")

    return upgrade_to_production


if __name__ == "__main__":
    epoch = 1
    num_workers = 4
    inf = float("inf")
    experiment_name = "Default"

    # Set the experiment name to an experiment in the shared experiments folder
    mlflow.set_experiment(experiment_name)

    # :TODO: use the same name and handle versions
    run_name = f"retrain_{int(time.time())}"
    alias_name = "chad"
    tag_key = "model_type"
    tag_value = "transformer"

    client = MlflowClient()

    # End any existing runs
    mlflow.end_run()

    # Start MLflow run for this experiment
    with mlflow.start_run(run_name=run_name) as run:
        (
            device,
            warmup,
            clip,
            scheduler,
            criterion,
            run_parameters,
            path_to_save_runs,
            train_loader,
            validation_loader,
            model,
            optimizer,
            latest_model,
            latest_version,
            latest_val_loss,
        ) = initialize_model_and_data(client, experiment_name, num_workers)
        mlflow.log_params(run_parameters)
        upgrade_to_production = run_training(
            path_to_save_runs=path_to_save_runs,
            device=device,
            total_epoch=epoch,
            best_loss=inf,
            train_loader=train_loader,
            validation_loader=validation_loader,
            model=model,
            warmup=warmup,
            clip=clip,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            latest_val_loss=latest_val_loss,
        )

    if upgrade_to_production:

        print("Registering model...")

        # Register the model
        model_version = mlflow.register_model(
            f"runs:/{run.info.run_id}/model", f"{run_name}"
        )

        # Transition the model version to the "production" stage
        client.transition_model_version_stage(
            name=run_name, version=model_version.version, stage="production"
        )

        # Add alias
        client.set_registered_model_alias(
            name=run_name, alias=alias_name, version=model_version.version
        )

        # Add tag
        client.set_model_version_tag(
            name=run_name, version=model_version.version, key=tag_key, value=tag_value
        )

    ### mlflow ui --host 0.0.0.0 --port 5000
