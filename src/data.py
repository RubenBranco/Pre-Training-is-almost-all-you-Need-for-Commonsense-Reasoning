
from tempfile import gettempdir
import os

from datasets import load_dataset
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm


def capitalize(s):
    return s[0].upper() + s[1:]


def uncapitalize(s):
    return s[0].lower() + s[1:]


def transform_operator(left_c, premise, mid_c, hypothesis, right_c):
    """
    left_c : Left conjunction
    mid_c : Middle conjunction
    right_c right conjunction
    """
    return "".join([left_c, premise, mid_c, hypothesis, right_c])


class CQADataMixin:
    """
    Mixin to encode CommonsenseQA dataset
    In this context, the premise is a question, the hypothesis are the
    multiple choice answers
    """

    LABELS = ["A", "B", "C", "D", "E"]
    NUM_HYPOTHESIS = len(LABELS)
    # Examples are prefixed with <s>, Q and :
    EXAMPLE_PREFIX_LEN = 3

    def encode_example(self, example, max_premise_size):
        input_ids = torch.zeros(
            [self.NUM_HYPOTHESIS, max_premise_size, self.hparams.max_seq_len],
            dtype=torch.int64,
        )
        attention_masks = torch.zeros(
            [self.NUM_HYPOTHESIS, max_premise_size, self.hparams.max_seq_len],
            dtype=torch.float32,
        )
        mlm_labels = torch.zeros(
            [self.NUM_HYPOTHESIS, max_premise_size, self.hparams.max_seq_len],
            dtype=torch.int64,
        )

        premise = self.tokenizer(
            example["question"],
            add_special_tokens=False,
            return_tensors="pt",
        )
        exs = []

        for i in range(self.NUM_HYPOTHESIS):
            exs.append(
                self.tokenizer(
                    transform_operator(
                        "Q: ",
                        example["question"],
                        " A: ",
                        example["choices"]["text"][i],
                        "",
                    ),
                    max_length=self.hparams.max_seq_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
            )

        p_tokens = self.tokenizer.pat.findall(example["question"])
        p_bpe_tokens = self.tokenizer.convert_ids_to_tokens(premise["input_ids"][0])
        bpe_offset_mapping = []

        cur_word = 0

        for i in range(len(p_bpe_tokens)):
            bpe_offset_mapping.append(i)
            if (
                self.tokenizer.convert_tokens_to_string(
                    p_bpe_tokens[bpe_offset_mapping[0] : i + 1]
                )
                == p_tokens[cur_word]
            ):
                start = bpe_offset_mapping[0] + self.EXAMPLE_PREFIX_LEN
                end = bpe_offset_mapping[-1] + 1 + self.EXAMPLE_PREFIX_LEN
                for j in range(self.NUM_HYPOTHESIS):
                    ex = exs[j]["input_ids"][0].clone()
                    original_ids_clone = ex[start:end].clone()
                    ex[start:end] = self.tokenizer.mask_token_id
                    input_ids[j, cur_word] = ex
                    attention_masks[j, cur_word] = exs[j]["attention_mask"][0].clone()
                    original_ids = torch.zeros(
                        self.hparams.max_seq_len, dtype=torch.float32
                    )
                    original_ids[start:end] = original_ids_clone
                    original_ids[original_ids == 0] = -100
                    mlm_labels[j, cur_word] = original_ids
                bpe_offset_mapping = []
                cur_word += 1

        return (
            input_ids,
            attention_masks,
            mlm_labels,
            self.LABELS.index(example["answerKey"]),
        )

    def encode_partition(self, examples):
        longest_premise = max(
            map(lambda x: len(self.tokenizer.pat.findall(x["question"])), examples)
        )
        input_ids = torch.zeros(
            [
                len(examples),
                self.NUM_HYPOTHESIS,
                longest_premise,
                self.hparams.max_seq_len,
            ],
            dtype=torch.int64,
        )
        attention_masks = torch.zeros(
            [
                len(examples),
                self.NUM_HYPOTHESIS,
                longest_premise,
                self.hparams.max_seq_len,
            ],
            dtype=torch.float32,
        )
        mlm_labels = torch.zeros(
            [
                len(examples),
                self.NUM_HYPOTHESIS,
                longest_premise,
                self.hparams.max_seq_len,
            ],
            dtype=torch.int64,
        )
        y = torch.zeros(len(examples), dtype=torch.int64)

        for i, example in tqdm(enumerate(examples), total=len(examples)):
            input_id, attention_mask, mlm_label, _y = self.encode_example(
                example, longest_premise
            )
            input_ids[i] = input_id
            attention_masks[i] = attention_mask
            mlm_labels[i] = mlm_label
            y[i] = _y

        return TensorDataset(input_ids, attention_masks, mlm_labels, y)

    def filter_premise_size(self, ds):
        return ds.filter(
            lambda ex: len(self.tokenizer.pat.findall(ex["question"]))
            <= self.hparams.premise_max_len
        )

    def get_cached_train_data_path(self):
        return os.path.join(
            gettempdir(), f"cqa_train_data_{self.hparams.model_name}.pt"
        )

    def get_cached_val_data_path(self):
        return os.path.join(gettempdir(), f"cqa_val_data_{self.hparams.model_name}.pt")

    def load_or_encode_data(self, ds):
        if not self.use_cached_data or not os.path.isfile(
            self.get_cached_train_data_path()
        ):
            self.train_data = self.encode_partition(
                self.filter_premise_size(ds["train"])
            )
            torch.save(self.train_data, self.get_cached_train_data_path())
        else:
            self.train_data = torch.load(self.get_cached_train_data_path())

        if not self.use_cached_data or not os.path.isfile(
            self.get_cached_val_data_path()
        ):
            self.val_data = self.encode_partition(ds["validation"])
            torch.save(self.val_data, self.get_cached_val_data_path())
        else:
            self.val_data = torch.load(self.get_cached_val_data_path())
        self.test_data = self.val_data

    def prepare_data(self):
        ds = load_dataset("commonsense_qa")
        self.load_or_encode_data(ds)

    def get_max_sentence_sizes(self):
        ds = load_dataset("commonsense_qa")
        out = {}

        for partition in ["train", "validation"]:
            _max = 0
            for example in ds[partition]:
                for i in range(self.NUM_HYPOTHESIS):
                    x = self.tokenizer(
                        transform_operator(
                            "Q: ",
                            example["question"],
                            " A: ",
                            example["choices"]["text"][i],
                            "",
                        ),
                        return_tensors="pt",
                    )
                    if x["input_ids"].size()[1] > _max:
                        _max = x["input_ids"].size()[1]
            out[partition] = _max

        return out


class ARCTDataMixin:
    """
    Mixin to encode ARCT dataset.
    """

    LABELS = ["warrant0", "warrant1"]
    NUM_HYPOTHESIS = len(LABELS)

    def transform_example(self, row, warrant_n):
        return (
            f"{capitalize(row['reason'].rstrip('.'))} and "
            f"{uncapitalize(row['warrant' + str(warrant_n)])} therefore "
        ) + row["claim"]

    def encode_example(self, example, max_premise_size):
        input_ids = torch.zeros(
            [self.NUM_HYPOTHESIS, max_premise_size, self.hparams.max_seq_len],
            dtype=torch.int64,
        )
        attention_masks = torch.zeros(
            [self.NUM_HYPOTHESIS, max_premise_size, self.hparams.max_seq_len],
            dtype=torch.float32,
        )
        mlm_labels = torch.zeros(
            [self.NUM_HYPOTHESIS, max_premise_size, self.hparams.max_seq_len],
            dtype=torch.int64,
        )

        exs = []

        for i in range(self.NUM_HYPOTHESIS):
            exs.append(
                self.tokenizer(
                    self.transform_example(example, i),
                    max_length=self.hparams.max_seq_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
            )

        reason = self.tokenizer(
            capitalize(example["reason"].rstrip(".")),
            add_special_tokens=False,
            return_tensors="pt",
        )
        claim = self.tokenizer(
            " " + example["claim"],
            add_special_tokens=False,
            return_tensors="pt",
        )
        warrants = {
            0: self.tokenizer(
                f" and {uncapitalize(example['warrant0'])} therefore",
                add_special_tokens=False,
                return_tensors="pt",
            ),
            1: self.tokenizer(
                f" and {uncapitalize(example['warrant1'])} therefore",
                add_special_tokens=False,
                return_tensors="pt",
            ),
        }

        reason_tokens = self.tokenizer.pat.findall(
            capitalize(example["reason"].rstrip("."))
        )
        reason_bpe_tokens = self.tokenizer.convert_ids_to_tokens(reason["input_ids"][0])

        claim_tokens = self.tokenizer.pat.findall(" " + example["claim"])
        claim_bpe_tokens = self.tokenizer.convert_ids_to_tokens(claim["input_ids"][0])

        bpe_offset_mapping = []

        cur_word = 0

        for i in range(len(reason_bpe_tokens)):
            bpe_offset_mapping.append(i)
            if (
                self.tokenizer.convert_tokens_to_string(
                    reason_bpe_tokens[bpe_offset_mapping[0] : i + 1]
                )
                == reason_tokens[cur_word]
            ):
                start = bpe_offset_mapping[0] + 1
                end = bpe_offset_mapping[-1] + 2
                for j in range(self.NUM_HYPOTHESIS):
                    ex = exs[j]["input_ids"][0].clone()
                    original_ids_clone = ex[start:end].clone()
                    ex[start:end] = self.tokenizer.mask_token_id
                    input_ids[j, cur_word] = ex
                    attention_masks[j, cur_word] = exs[j]["attention_mask"][0].clone()
                    original_ids = torch.zeros(
                        self.hparams.max_seq_len, dtype=torch.float32
                    )
                    original_ids[start:end] = original_ids_clone
                    original_ids[original_ids == 0] = -100
                    mlm_labels[j, cur_word] = original_ids
                bpe_offset_mapping = []
                cur_word += 1

        bpe_offset_mapping = []

        cur_claim_word = 0

        for i in range(len(claim_bpe_tokens)):
            bpe_offset_mapping.append(i)
            if (
                self.tokenizer.convert_tokens_to_string(
                    claim_bpe_tokens[bpe_offset_mapping[0] : i + 1]
                )
                == claim_tokens[cur_claim_word]
            ):
                reason_size = reason["input_ids"].size()[1]
                for j in range(self.NUM_HYPOTHESIS):
                    warrant_size = warrants[j]["input_ids"].size()[1]
                    start = bpe_offset_mapping[0] + 1 + reason_size + warrant_size
                    end = bpe_offset_mapping[-1] + 2 + reason_size + warrant_size
                    ex = exs[j]["input_ids"][0].clone()
                    original_ids_clone = ex[start:end].clone()
                    ex[start:end] = self.tokenizer.mask_token_id
                    input_ids[j, cur_word] = ex
                    attention_masks[j, cur_word] = exs[j]["attention_mask"][0].clone()
                    original_ids = torch.zeros(
                        self.hparams.max_seq_len, dtype=torch.float32
                    )
                    original_ids[start:end] = original_ids_clone
                    original_ids[original_ids == 0] = -100
                    mlm_labels[j, cur_word] = original_ids
                bpe_offset_mapping = []
                cur_word += 1
                cur_claim_word += 1

        return (
            input_ids,
            attention_masks,
            mlm_labels,
            int(example["correctLabelW0orW1"]),
        )

    def get_premise_or_hyp_max_length(self, df):
        return df["premise_length"].max()

    def encode_partition(self, df):
        longest_premise = self.get_premise_or_hyp_max_length(df)
        input_ids = torch.zeros(
            [
                len(df),
                self.NUM_HYPOTHESIS,
                longest_premise,
                self.hparams.max_seq_len,
            ],
            dtype=torch.int64,
        )
        attention_masks = torch.zeros(
            [
                len(df),
                self.NUM_HYPOTHESIS,
                longest_premise,
                self.hparams.max_seq_len,
            ],
            dtype=torch.float32,
        )
        mlm_labels = torch.zeros(
            [
                len(df),
                self.NUM_HYPOTHESIS,
                longest_premise,
                self.hparams.max_seq_len,
            ],
            dtype=torch.int64,
        )
        y = torch.zeros(len(df), dtype=torch.int64)

        for i, example in tqdm(df.iterrows(), total=len(df)):
            input_id, attention_mask, mlm_label, _y = self.encode_example(
                example, longest_premise
            )
            input_ids[i] = input_id
            attention_masks[i] = attention_mask
            mlm_labels[i] = mlm_label
            y[i] = _y

        return TensorDataset(input_ids, attention_masks, mlm_labels, y)

    def filter_premise_size(self, df):
        return df[df["premise_length"] <= self.hparams.premise_max_len].reset_index()

    def get_cached_train_data_path(self):
        return os.path.join(
            gettempdir(), f"arct_train_data_{self.hparams.model_name}.pt"
        )

    def get_cached_val_data_path(self):
        return os.path.join(gettempdir(), f"arct_val_data_{self.hparams.model_name}.pt")

    def get_cached_test_data_path(self):
        return os.path.join(
            gettempdir(), f"arct_test_data_{self.hparams.model_name}.pt"
        )

    def get_train_df(self):
        return pd.read_csv(os.path.join(self.data_path, "train.csv"), sep="\t")

    def get_val_df(self):
        return pd.read_csv(os.path.join(self.data_path, "dev.csv"), sep="\t")

    def get_test_df(self):
        return pd.read_csv(os.path.join(self.data_path, "test.csv"), sep="\t")

    def calculate_prem_and_hyp_length(self, df):
        df["premise_length"] = [
            len(self.tokenizer.pat.findall(capitalize(r["reason"].rstrip("."))))
            + len(self.tokenizer.pat.findall(r["claim"]))
            for _, r in df.iterrows()
        ]
        return df

    def load_or_encode_data(self):
        if not self.use_cached_data or not os.path.isfile(
            self.get_cached_train_data_path()
        ):
            df = self.calculate_prem_and_hyp_length(self.get_train_df())
            self.train_data = self.encode_partition(self.filter_premise_size(df))
            torch.save(self.train_data, self.get_cached_train_data_path())
        else:
            self.train_data = torch.load(self.get_cached_train_data_path())

        if not self.use_cached_data or not os.path.isfile(
            self.get_cached_val_data_path()
        ):
            df = self.calculate_prem_and_hyp_length(self.get_val_df())
            self.val_data = self.encode_partition(df)
            torch.save(self.val_data, self.get_cached_val_data_path())
        else:
            self.val_data = torch.load(self.get_cached_val_data_path())

        if not self.use_cached_data or not os.path.isfile(
            self.get_cached_test_data_path()
        ):
            df = self.calculate_prem_and_hyp_length(self.get_test_df())
            self.test_data = self.encode_partition(df)
            torch.save(self.val_data, self.get_cached_val_data_path())
        else:
            self.test_data = torch.load(self.get_cached_test_data_path())

    def prepare_data(self):
        self.load_or_encode_data()

    def get_max_sentence_sizes(self):
        dfs = {
            "train": self.get_train_df(),
            "val": self.get_val_df(),
            "test": self.get_test_df(),
        }
        out = {}

        for partition in dfs:
            _max = 0
            _max_id = 0
            for i, example in dfs[partition].iterrows():
                for warrant_n in range(self.NUM_HYPOTHESIS):
                    x = self.tokenizer(
                        self.transform_example(example, warrant_n),
                        return_tensors="pt",
                    )
                    if x["input_ids"].size()[1] > _max:
                        _max = x["input_ids"].size()[1]
                        _max_id = i
            out[partition] = (_max, _max_id)

        return out
