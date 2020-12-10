import os

import click

from train_and_eval import train_and_eval_cqa, train_and_eval_arct


@click.group()
def cli():
    pass


@click.command()
@click.option("--do_train", is_flag=True, help="Train the model.")
@click.option("--do_test", is_flag=True, help="Test the model.")
@click.option(
    "--load_from_checkpoint",
    default=None,
    type=click.Path(),
    help="Whether to load the model from a given checkpoint. (default: None)",
)
@click.option(
    "--epochs",
    default=10,
    type=int,
    help="Epochs of training (default: 10)",
)
@click.option(
    "--learning_rate",
    default=1e-5,
    type=float,
    help="Learning rate (default: 1e-5)",
)
@click.option(
    "--weight_decay",
    default=0.1,
    type=float,
    help="Weight Decay for Adam opt (default: 0.1)",
)
@click.option(
    "--warmup_ratio",
    default=0.06,
    type=float,
    help="Warmup ratio for linear schedule (default: 0.06)",
)
@click.option(
    "--loss_threshold",
    default=0.5,
    type=float,
    help="Margin loss threshold (default: 0.5)",
)
@click.option(
    "--gradient_accumulation_steps",
    default=0,
    type=int,
    help="How many batches should the gradient accumulate for before doing an update. (default: 0)",
)
@click.option(
    "--seed",
    default=42,
    type=int,
    help="Random seed. (default: 42)",
)
@click.option(
    "--model_name",
    default="roberta-base",
    type=str,
    help="Model name from huggingface roberta pre-trained models. (default: roberta-base)",
)
@click.option(
    "--batch_size", default=8, type=int, help="Training batch size. (default: 8)"
)
@click.option(
    "--forward_pass_size",
    default=8,
    type=int,
    help="How many examples per <p, hi> forward pass (default: 8)",
)
@click.option(
    "--max_seq_len",
    default=80,
    type=int,
    help="Maximum length for any example sequence (in subwords). (default: 80)",
)
@click.option(
    "--premise_max_len",
    default=30,
    type=int,
    help=(
        "Maximum length allowed for a premise in the training data "
        "(if bigger, gets left out) in subwords. (default: 30)"
    ),
)
@click.option(
    "--log_path",
    default=os.path.join(os.getcwd(), "logs"),
    type=click.Path(),
    help="Path to directory where tensorboard logs will be saved. (default: ./logs/)",
)
@click.option(
    "--checkpoint_path",
    default=os.path.join(os.getcwd(), "checkpoints"),
    type=click.Path(),
    help="Path to directory where model checkpoints will be saved. (default: ./checkpoints/)",
)
@click.option(
    "--use_cached_data",
    is_flag=True,
    help="Use preprocessed cached data if exists",
)
@click.option(
    "--save_top_k",
    default=-1,
    type=int,
    help=(
        "Save top k model checkpoints, if -1 saves all, else saves the input number, "
        "according to validation loss. (default: -1)"
    )
)
@click.option(
    "--use_early_stop",
    is_flag=True,
    help="Whether to use early stopping.",
)
@click.option(
    "--early_stop_metric",
    type=str,
    default="Loss/Validation",
    help="Metric to monitor for early stopping criteria. (default: Loss/Validation)",
)
@click.option(
    "--early_stop_patience",
    type=int,
    default=3,
    help=(
        "Number of validation epochs with no improvement after which training will be "
        "stopped. (default: 3)"
    ),
)
@click.option(
    "--early_stop_mode",
    type=click.Choice(["auto", "min", "max"]),
    default="auto",
    help=(
        "Min mode, training will stop when the quantity monitored has stopped decreasing;"
        "in max mode it will stop when the quantity monitored has stopped increasing;"
    )
)
def plausibility_roberta_cqa(
    do_train,
    do_test,
    load_from_checkpoint,
    epochs,
    learning_rate,
    weight_decay,
    warmup_ratio,
    loss_threshold,
    gradient_accumulation_steps,
    seed,
    model_name,
    batch_size,
    forward_pass_size,
    max_seq_len,
    premise_max_len,
    log_path,
    checkpoint_path,
    use_cached_data,
    save_top_k,
    use_early_stop,
    early_stop_metric,
    early_stop_patience,
    early_stop_mode,
):
    train_and_eval_cqa(
        do_train,
        do_test,
        load_from_checkpoint,
        epochs,
        learning_rate,
        weight_decay,
        warmup_ratio,
        loss_threshold,
        gradient_accumulation_steps,
        seed,
        model_name,
        batch_size,
        forward_pass_size,
        max_seq_len,
        premise_max_len,
        log_path,
        checkpoint_path,
        use_cached_data,
        save_top_k,
        use_early_stop,
        early_stop_metric,
        early_stop_patience,
        early_stop_mode,
    )


@click.command()
@click.option("--do_train", is_flag=True, help="Train the model.")
@click.option("--do_test", is_flag=True, help="Test the model.")
@click.option(
    "--load_from_checkpoint",
    default=None,
    type=click.Path(),
    help="Whether to load the model from a given checkpoint. (default: None)",
)
@click.option(
    "--epochs",
    default=10,
    type=int,
    help="Epochs of training (default: 10)",
)
@click.option(
    "--learning_rate",
    default=1e-5,
    type=float,
    help="Learning rate (default: 1e-5)",
)
@click.option(
    "--weight_decay",
    default=0.1,
    type=float,
    help="Weight Decay for Adam opt (default: 0.1)",
)
@click.option(
    "--warmup_ratio",
    default=0.06,
    type=float,
    help="Warmup ratio for linear schedule (default: 0.06)",
)
@click.option(
    "--loss_threshold",
    default=0.5,
    type=float,
    help="Margin loss threshold (default: 0.5)",
)
@click.option(
    "--gradient_accumulation_steps",
    default=0,
    type=int,
    help="How many batches should the gradient accumulate for before doing an update. (default: 0)",
)
@click.option(
    "--seed",
    default=42,
    type=int,
    help="Random seed. (default: 42)",
)
@click.option(
    "--model_name",
    default="roberta-base",
    type=str,
    help="Model name from huggingface roberta pre-trained models. (default: roberta-base)",
)
@click.option(
    "--batch_size", default=8, type=int, help="Training batch size. (default: 8)"
)
@click.option(
    "--forward_pass_size",
    default=8,
    type=int,
    help="How many examples per <p, hi> forward pass (default: 8)",
)
@click.option(
    "--max_seq_len",
    default=80,
    type=int,
    help="Maximum length for any example sequence (in subwords). (default: 80)",
)
@click.option(
    "--premise_max_len",
    default=30,
    type=int,
    help=(
        "Maximum length allowed for a premise in the training data "
        "(if bigger, gets left out) in subwords. (default: 30)"
    ),
)
@click.option(
    "--data_path",
    default=os.path.join(os.getcwd(), "data"),
    type=click.Path(),
    help="Path to directory where data is located. (default: ./data/)",
)
@click.option(
    "--log_path",
    default=os.path.join(os.getcwd(), "logs"),
    type=click.Path(),
    help="Path to directory where tensorboard logs will be saved. (default: ./logs/)",
)
@click.option(
    "--checkpoint_path",
    default=os.path.join(os.getcwd(), "checkpoints"),
    type=click.Path(),
    help="Path to directory where model checkpoints will be saved. (default: ./checkpoints/)",
)
@click.option(
    "--use_cached_data",
    is_flag=True,
    help="Use preprocessed cached data if exists",
)
@click.option(
    "--save_top_k",
    default=-1,
    type=int,
    help=(
        "Save top k model checkpoints, if -1 saves all, else saves the input number, "
        "according to validation loss. (default: -1)"
    )
)
@click.option(
    "--use_early_stop",
    is_flag=True,
    help="Whether to use early stopping.",
)
@click.option(
    "--early_stop_metric",
    type=str,
    default="Loss/Validation",
    help="Metric to monitor for early stopping criteria. (default: Loss/Validation)",
)
@click.option(
    "--early_stop_patience",
    type=int,
    default=3,
    help=(
        "Number of validation epochs with no improvement after which training will be "
        "stopped. (default: 3)"
    ),
)
@click.option(
    "--early_stop_mode",
    type=click.Choice(["auto", "min", "max"]),
    default="auto",
    help=(
        "Min mode, training will stop when the quantity monitored has stopped decreasing;"
        "in max mode it will stop when the quantity monitored has stopped increasing;"
    )
)
def plausibility_roberta_arct(
    do_train,
    do_test,
    load_from_checkpoint,
    epochs,
    learning_rate,
    weight_decay,
    warmup_ratio,
    loss_threshold,
    gradient_accumulation_steps,
    seed,
    model_name,
    batch_size,
    forward_pass_size,
    max_seq_len,
    premise_max_len,
    data_path,
    log_path,
    checkpoint_path,
    use_cached_data,
    save_top_k,
    use_early_stop,
    early_stop_metric,
    early_stop_patience,
    early_stop_mode,
):
    train_and_eval_arct(
        do_train,
        do_test,
        load_from_checkpoint,
        epochs,
        learning_rate,
        weight_decay,
        warmup_ratio,
        loss_threshold,
        gradient_accumulation_steps,
        seed,
        model_name,
        batch_size,
        forward_pass_size,
        max_seq_len,
        premise_max_len,
        data_path,
        log_path,
        checkpoint_path,
        use_cached_data,
        save_top_k,
        use_early_stop,
        early_stop_metric,
        early_stop_patience,
        early_stop_mode,
    )


if __name__ == "__main__":
    cli.add_command(plausibility_roberta_cqa)
    cli.add_command(plausibility_roberta_arct)
    cli()
