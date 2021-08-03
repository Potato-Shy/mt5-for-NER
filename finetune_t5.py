import os
import shutil
from datetime import datetime
from typing import Generator
from typing import Iterator
import argparse
from absl import app
import yaml
from logging import Logger
import logging
import json

import torch

from torch.utils.data import DataLoader, Dataset
from transformers import MT5ForConditionalGeneration
from transformers import MT5Tokenizer
from transformers import MT5Config

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from tqdm import tqdm

EXPERIMENT_CONFIG_NAME = "config.yml"

parser = argparse.ArgumentParser()

parser.add_argument("--project_config_path", type=str, default="./project_config.yml")

args = parser.parse_args()

SEPARATOR = " | "

def create_logger(log_file: str) -> Logger:
    """
    Create logger for logging the experiment process.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level=logging.DEBUG)
    logger.addHandler(file_handler)
    file_handler.setFormatter(formatter)

    return logger


def load_config(file_name):
    conf_file = yaml.full_load(open(file_name, "r"))
    return conf_file


def formate_output(output_s):
    format_res = []
    for type_entity in output_s:
        type_, entity_txt = type_entity.split(":")
        for entity in entity_txt.split(","):
            format_res.append("-".join([type_, entity]))
    return format_res


class NaiveDateset(Dataset):

    def __init__(self, data_path, max_input_length, max_output_length):

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.data = self.load_data(data_path)

    def load_data(self, data_path, convert_to_ids=True):
        with open(data_path, "r", encoding="utf-8") as reader:
            data = []
            for line in reader:
                act, utterance = line.strip().split("\t")[:2]
                if convert_to_ids:
                    act_input_ = tokenizer(act, max_length=self.max_input_length, padding="max_length", truncation=True)
                    act_input_ids = act_input_["input_ids"]
                    attention_mask = act_input_["attention_mask"]
                    utterance_input_ = tokenizer(utterance, max_length=self.max_output_length,
                                                           padding="max_length", truncation=True)
                    utterance_input_ids = utterance_input_["input_ids"]
                    utterance_mask = utterance_input_["attention_mask"]
                    data.append((act_input_ids, attention_mask, utterance_input_ids, utterance_mask))
                else:
                    data.append((act, utterance))
        return data

    def __getitem__(self, data_index):
        return self.data[data_index]

    def __len__(self):
        return len(self.data)


class MT5_model:
    """
    This is a trainer class for finetuning Huggingface T5 implementation on a parallel dataset.

    Attributes:
        model_config: model configuration
        train_config: training configuration
        data_config: data paths configuration
        experiment_path: path where experiment output will be dumped
        tokenizer: tokenizer
        device: device where experiment will happen (gpu or cpu)
        logger: File and terminal logger
    """

    def __init__(self, config: dict, timestamp=None):

        self.data_config = config["data"]
        self.model_config = config["model"]
        self.train_config = config["training"]
        if timestamp:
            self.logger = create_logger("./logger/log-{}.txt".format(timestamp))
        else:
            self.logger = create_logger("./logger/log.txt")
            self.logger.info("created log.txt in ./logger/")

        self.device = torch.device(
            "cuda" if self.train_config["use_cuda"] and torch.cuda.is_available() else "cpu"
        )
        self.rouge = Rouge()

        self.logger.info(f"Experiment Output Path: \n ./logger/")
        self.logger.info(f"Training will be begin with this configuration: \n {config} ")

    def build_model(self) -> None:
        """
        Build model and update its configuration.
        """
        # self.model = MT5ForConditionalGeneration.from_pretrained(self.model_config["model_file"])

        model_config = MT5Config.from_json_file(self.model_config["model_config"])
        self.model = MT5ForConditionalGeneration(model_config)
        model_state_dict = torch.load(self.model_config["model_file"])
        try:
            self.model.load_state_dict(model_state_dict)
        except:
            if model_config.vocab_size != self.train_config["vocab_size"]:
                self.model.resize_token_embeddings(self.train_config["vocab_size"])
                self.model.load_state_dict(model_state_dict)
            else:
                raise Exception("load model wrong")
        # model_config = {
        #     "early_stopping": self.train_config["early_stopping"],
        #     "max_length": self.train_config["max_output_length"],
        #     "num_beams": self.train_config["beam_size"],
        #     "prefix": self.data_config["src_prefix"],
        #     "vocab_size": self.tokenizer.vocab_size,
        # }
        # self.model.config.update(model_config)
        self.model.to(self.device)

    def _build_optimizer(self, model_parameters: Iterator) -> torch.optim.Optimizer:
        """
        Build optimizer to be used in training.
        """
        if self.train_config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                model_parameters,
                weight_decay=self.train_config["weight_decay"],
                lr=self.train_config["learning_rate"],
            )
        elif self.train_config["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                model_parameters,
                weight_decay=self.train_config["weight_decay"],
                lr=self.train_config["learning_rate"],
            )
        else:
            self.logger.warning(
                "Only 'adam' and 'sgd' is currently supported. Will use adam as default"
            )
            optimizer = torch.optim.Adam(
                model_parameters,
                weight_decay=self.train_config["weight_decay"],
                lr=self.train_config["learning_rate"],
            )
        return optimizer

    def _create_datasets(self) -> tuple:
        def _collate_fn(batch):
            #  batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
            batch = list(zip(*batch))
            act = torch.tensor(batch[0], dtype=torch.int64)
            attention_mask = torch.tensor(batch[1], dtype=torch.int64)
            utterance = torch.tensor(batch[2], dtype=torch.int64)
            utterance_mask = torch.tensor(batch[3], dtype=torch.int64)
            del batch
            return act, attention_mask, utterance, utterance_mask

        if self.train_config["mode"] == "train":
            train_dataset = NaiveDateset(
                self.data_config["train_file"],
                self.train_config["max_input_length"],
                self.train_config["max_output_length"]
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.train_config["batch_size"],
                shuffle=True,
                num_workers=self.train_config["num_workers"],
                drop_last=False,
                collate_fn=_collate_fn
            )
        else:
            train_dataloader = None

        if self.train_config["mode"] == "train" or self.train_config["mode"] == "dev":
            dev_dataset = NaiveDateset(
                self.data_config["dev_file"],
                self.train_config["max_input_length"],
                self.train_config["max_output_length"]
            )
            dev_dataloader = DataLoader(
                dev_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.train_config["num_workers"],
                drop_last=False,
                collate_fn=_collate_fn
            )
            self.dev_data = dev_dataset.load_data(self.data_config["dev_file"], convert_to_ids=False)
            test_dataloader = None
        else:
            dev_dataloader = None
            test_dataset = NaiveDateset(
                self.data_config["test_file"],
                self.model_config["max_input_length"],
                self.model_config["max_output_length"]
            )
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self.train_config["num_workers"],
                drop_last=False,
                collate_fn=_collate_fn
            )
        return train_dataloader, dev_dataloader, test_dataloader

    def evaluate(self, dev_dataloader):
        smooth = SmoothingFunction().method1
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
        total = 0
        print(f"===== Start Evaluate =====\n")
        self.model.eval()
        with torch.no_grad():
            for i, (input_ids, _, reference_ids, _) in enumerate(tqdm(dev_dataloader)):
                total += 1
                input_ids, reference_ids = input_ids.to(self.device), reference_ids.to(self.device)
                output_ids = self.model.generate(input_ids)
                output_s = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                output_s = output_s.split(SEPARATOR)
                predict_res = formate_output(output_s)


        return

    def train(self):
        self.best_bleu = self.train_config["best_bleu"]
        train_dataloader, dev_dataloader, _ = self._create_datasets()
        total_step = len(train_dataloader)
        optimizer = self._build_optimizer(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "min",
            self.train_config["reduction_factor"],
            self.train_config["patience"],
            verbose=True,
            min_lr=self.train_config["min_lr"],
        )

        print(f"===== Start Training =====\n")

        for epoch in range(self.train_config["epochs"]):
            self.model.train()
            epoch_loss = 0.0
            step = 0
            with tqdm(total=len(train_dataloader), ncols=120) as _tqdm:
                _tqdm.set_description("Training epoch: {}/{}".format(epoch + 1, self.train_config["epochs"]))
                for input_ids, attention_mask, reference_ids, reference_mask in train_dataloader:
                    step += 1
                    # input_ids, attention_mask, reference_ids = input_ids.to(self.device), attention_mask.to(self.device), reference_ids.to(self.device)

                    outputs = self.model(input_ids=input_ids.to(self.device),
                                         attention_mask=attention_mask.to(self.device),
                                         labels=reference_ids.to(self.device),
                                         decoder_attention_mask=reference_mask.to(self.device)
                                         )
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    #                    scheduler.step(100)

                    optimizer.zero_grad()
                    epoch_loss += loss.item()
                    _tqdm.set_postfix(loss=" {:.6f}".format(epoch_loss / step))
                    _tqdm.update()
                    # tqdm.write(
                    #     "Epoch: %d, Step: %5d, loss: %.3f"
                    #     % (epoch + 1, step, epoch_loss / step)
                    # )
                self.logger.info(
                    f"Epoch {epoch + 1} done. Average Loss: {epoch_loss / step}"
                )
                print(f"Epoch {epoch + 1} done. Average Loss: {epoch_loss / step}")
            if (epoch + 1) % self.train_config["evaluate_epochs"] == 0 or \
                    epoch + 1 == self.train_config["epochs"]:
                metrics = self.evaluate(dev_dataloader)
                if metrics['bleu'] > self.best_bleu:
                    self.best_bleu = metrics['bleu']
                    torch.save(self.model.state_dict(),
                               self.model_config["output_model_file"] + "best_model_{:.5f}.pkl".format(
                                   self.best_bleu))  # 保存模型
                metrics['best_bleu'] = self.best_bleu
                print('valid_data:', metrics)
                self.logger.info(f"valid_data: {metrics}")

    def predict(self, text):
        # self.model.forward()
        _max_input_length = self.train_config["max_input_length"]

        input_ids = tokenizer.encode(text, max_length=_max_input_length, padding="max_length", return_tensors="pt", truncation=True)
        input_ids = input_ids.to(self.device)
        self.model.eval()
        output = self.model.generate(input_ids)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        return output


if __name__ == "__main__":
    # load config
    config = load_config(args.project_config_path)

    # specify and create experiment path
    timestamp = datetime.now().strftime("_%Y%m%d-%H%M%S")
    if config["model"]["tokenizer_path"]:
        tokenizer = MT5Tokenizer.from_pretrained(config["model"]["tokenizer_path"])
    else:
        raise ValueError("tokenizer path cant be none!")
    # initialize trainer and train

    MT5 = MT5_model(config, timestamp=timestamp)
    MT5.build_model()
    if config["training"]["mode"] == "train":
        MT5.train()
    if config["training"]["mode"] == "predict":
        texts = [
            "SYSTEM | service_name=餐馆 | inform,餐馆-名称, values = 麻辣人生 | nooffer,餐馆-none",
            "SYSTEM | service_name=景点 | inform,景点-名称, values = 八达岭长城 | inform,景点-评分, values = 4.6分 | inform,景点-门票, values = 35元",
            "SYSTEM | service_name=景点 | inform,景点-电话, values = 010-67389898,010-67383333,010-67201818",
            "SYSTEM | service_name=餐馆 | recommend,餐馆-名称, values = 东来顺(前门大街店) | recommend,餐馆-名称, values = 全聚德烤鸭店(清华园店)",
            "SYSTEM | service_name=餐馆 | inform,餐馆-周边餐馆, values = 北京全聚德(王府井店) | inform,餐馆-周边餐馆, values = 北新桥卤煮老店 | inform,餐馆-周边餐馆, values = 姚记炒肝店（鼓楼店） | inform,餐馆-周边餐馆, values = 鬼味烤翅",
            "SYSTEM | frame = None | last = 哦了，我记下了，谢谢你啊",
            "SYSTEM | service_name=景点 | inform,景点-名称, values = 北京林业大学"
        ]
        for text in texts:
            print(MT5.predict(text))
        print("=====")
        for text in texts:
            print(MT5.predict(text))



