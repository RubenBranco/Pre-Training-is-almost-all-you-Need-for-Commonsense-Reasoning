from argparse import Namespace
import logging
import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


from models import (
    PlausibilityRankingRoBERTaForCQA,
    PlausibilityRankingRoBERTaForARCT,
)


def train_and_eval_cqa(
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
    hparams = Namespace(
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
        model_name=model_name,
        batch_size=batch_size,
        forward_pass_size=forward_pass_size,
        max_seq_len=max_seq_len,
        premise_max_len=premise_max_len,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        loss_threshold=loss_threshold,
    )
    seed_everything(hparams.seed)

    if load_from_checkpoint is None:
        roberta = PlausibilityRankingRoBERTaForCQA(
            hparams, epochs=epochs, use_cached_data=use_cached_data
        )
    else:
        roberta = PlausibilityRankingRoBERTaForCQA.load_from_checkpoint(
            load_from_checkpoint,
            hparams=hparams,
            epochs=epochs,
            use_cached_data=use_cached_data
        )
    roberta_name = "Plausibility_RoBERTa_CQA"

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, roberta_name + "_{epoch:02d}"),
        save_top_k=save_top_k,
        monitor="Loss/Validation",
        mode="min",
    )
    logger = TensorBoardLogger(log_path, name=roberta_name)

    kwargs = dict()

    if hparams.gradient_accumulation_steps:
        kwargs["accumulate_grad_batches"] = hparams.gradient_accumulation_steps

    if use_early_stop:
        kwargs["callbacks"] = [
            EarlyStopping(
                monitor=early_stop_metric,
                patience=early_stop_patience,
                mode=early_stop_mode,
            )
        ]

    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        gpus=1,
        **kwargs,
    )

    if do_train:
        trainer.fit(roberta)
    if do_test:
        trainer.test(roberta)


def train_and_eval_arct(
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
    hparams = Namespace(
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        seed=seed,
        model_name=model_name,
        batch_size=batch_size,
        forward_pass_size=forward_pass_size,
        max_seq_len=max_seq_len,
        premise_max_len=premise_max_len,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        loss_threshold=loss_threshold,
    )
    seed_everything(hparams.seed)

    if load_from_checkpoint is None:
        roberta = PlausibilityRankingRoBERTaForARCT(
            hparams, data_path=data_path, epochs=epochs, use_cached_data=use_cached_data
        )
    else:
        roberta = PlausibilityRankingRoBERTaForARCT.load_from_checkpoint(
            load_from_checkpoint,
            hparams=hparams,
            data_path=data_path,
            epochs=epochs,
            use_cached_data=use_cached_data,
        )
    roberta_name = "Plausibility_RoBERTa_ARCT"

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, roberta_name + "_{epoch:02d}"),
        save_top_k=save_top_k,
        monitor="Loss/Validation",
        mode="min",
    )
    logger = TensorBoardLogger(log_path, name=roberta_name)
    kwargs = dict()

    if hparams.gradient_accumulation_steps:
        kwargs["accumulate_grad_batches"] = hparams.gradient_accumulation_steps

    if use_early_stop:
        kwargs["callbacks"] = [
            EarlyStopping(
                monitor=early_stop_metric,
                patience=early_stop_patience,
                mode=early_stop_mode,
            )
        ]

    trainer = Trainer(
        max_epochs=epochs,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        gpus=1,
        **kwargs
    )

    if do_train:
        trainer.fit(roberta)
    if do_test:
        trainer.test(roberta)
