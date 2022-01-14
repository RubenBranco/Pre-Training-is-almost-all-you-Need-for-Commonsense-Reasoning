import warnings
from os import cpu_count

import torch
import transformers
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, f1_score

from data import CQADataMixin, ARCTDataMixin
from loss import ssm_loss


"""
Adapted from HuggingFace source code.
"""
class RobertaForMaskedLM(transformers.RobertaForMaskedLM):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        reduce_loss=True,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            masked_lm_loss = loss_fct(prediction_scores.permute(0, 2, 1), labels).mean(
                dim=1
            )
            if reduce_loss:
                masked_lm_loss = masked_lm_loss.sum()

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return transformers.modeling_outputs.MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PlausibilityRankingRoBERTa(pl.LightningModule):
    def __init__(self, hparams, data_path=None, epochs=None, use_cached_data=True):
        super().__init__()
        self.hparams = hparams

        self.tokenizer = transformers.RobertaTokenizer.from_pretrained(
            hparams.model_name,
            cache_dir="/tmp/",
        )
        self.roberta = RobertaForMaskedLM.from_pretrained(
            hparams.model_name,
            cache_dir="/tmp/",
            return_dict=True,
        )
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.data_path = data_path
        self.epochs = epochs
        self.use_cached_data = use_cached_data
        self.batch_size = hparams.batch_size

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=cpu_count(),
        )

    def configure_optimizers(self):
        steps = len(self.train_dataloader()) * self.epochs
        opt = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            opt,
            steps * self.hparams.warmup_ratio,
            steps,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": scheduler,
        }

    def training_step(self, batch, _):
        input_ids, attention_masks, mlm_labels, y = batch
        loss, _, preds = self.forward(input_ids, attention_masks, mlm_labels, y=y)
        accuracy_score = accuracy(preds, y, num_classes=self.NUM_HYPOTHESIS)
        log = {"Loss/Train": loss, "Accuracy/Train": accuracy_score}
        self.log_dict(log, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, _):
        input_ids, attention_masks, mlm_labels, y = batch
        loss, _, preds = self.forward(input_ids, attention_masks, mlm_labels, y=y)
        return {
            "val_loss": loss,
            "val_tp": torch.sum(torch.eq(y, preds.detach()).view(-1)).item(),
            "batch_size": input_ids.shape[0],
        }

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.FloatTensor([x["val_loss"] for x in outputs]).mean()
        val_accuracy = torch.FloatTensor([x["val_tp"] for x in outputs])
        total_examples = (
            torch.FloatTensor([x["batch_size"] for x in outputs]).sum().item()
        )
        val_accuracy = torch.FloatTensor([val_accuracy.sum().item() / total_examples])
        self.log_dict(
            {
                "Loss/Validation": avg_val_loss,
                "Accuracy/Validation": val_accuracy,
            },
            prog_bar=True,
        )

    def test_step(self, batch, _):
        input_ids, attention_masks, mlm_labels, y = batch
        _, preds = self.forward(input_ids, attention_masks, mlm_labels)
        return {"y_pred": preds, "y": y}

    def test_epoch_end(self, outputs):
        y_pred = torch.cat([x["y_pred"] for x in outputs], dim=0)
        y = torch.cat([x["y"] for x in outputs], dim=0).flatten()
        accuracy_score = accuracy(y_pred, y, num_classes=self.NUM_HYPOTHESIS)
        f1 = f1_score(y_pred, y)
        log = {"Test/Accuracy": accuracy_score, "Test/f1": f1}
        self.log_dict(log, prog_bar=True)

    def forward(
        self,
        input_ids,
        attention_masks,
        mlm_labels,
        y=None,
    ):
        num_examples = input_ids.shape[0]
        num_hypothesis = input_ids.shape[1]
        ssm = torch.zeros([num_examples, num_hypothesis], dtype=torch.float32).to(
            self.device
        )
        preds = torch.zeros(num_examples, dtype=torch.int32).to(self.device)

        for i in range(num_examples):
            for j in range(num_hypothesis):
                idxs = torch.nonzero(
                    (input_ids[i, j, :].sum(dim=1) != 0).long()
                ).flatten()
                prem_score = torch.zeros(idxs.shape[0]).to(self.device)

                for cur_idxs in idxs.split(self.hparams.forward_pass_size):
                    outputs = self.roberta(
                        input_ids=input_ids[i, j, cur_idxs],
                        attention_mask=attention_masks[i, j, cur_idxs],
                        labels=mlm_labels[i, j, cur_idxs],
                    )
                    prem_score[cur_idxs] = outputs["loss"]
                ssm[i, j] = prem_score.sum()
            preds[i] = ssm[i].argmin()

        outs = (
            ssm,
            preds,
        )

        if y is not None:
            outs = (
                ssm_loss(ssm, y, self.hparams.loss_threshold),
                *outs,
            )
        return outs

    def prepare_data(self):
        raise NotImplementedError()


class PlausibilityRankingRoBERTaForCQA(CQADataMixin, PlausibilityRankingRoBERTa):
    pass


class PlausibilityRankingRoBERTaForARCT(ARCTDataMixin, PlausibilityRankingRoBERTa):
    pass
