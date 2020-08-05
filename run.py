
import os
import sys
import warnings

import argparse
import yaml

from tqdm import tqdm

import torch
from torch.nn import CrossEntropyLoss
import pandas as pd

sys.path.append("./src")

import optimizer
import model
import data

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-S", '--setting', type=str, default="cifar10+BC_k12")
    parser.add_argument("-M", '--mode', type=str,
                        default="train", choices=["train", "test"])
    args = parser.parse_args()
    with open("./setting.yml", "r") as f:
        configs = yaml.safe_load(f)
    config = configs[args.setting]
    config["save_dir"] = f"./work/{args.setting}"
    config["mode"] = args.mode
    if os.path.isdir(config["save_dir"]) and config["mode"] == "train":
        warnings.warn(
            "directory:{} is already exists".format(config['save_dir']))
    else:
        os.makedirs(config["save_dir"])
    return config


def _train(network, criterion, trainLoader, device, optimizer):
    network.train()
    epoch_loss = 0
    epoch_acc = 0
    num_data = 0
    for input_x, target_y in tqdm(trainLoader):
        input_x, target_y = input_x.to(device), target_y.to(device)
        optimizer.zero_grad()
        predict_y = network(input_x)
        loss = criterion(predict_y, target_y)
        loss.backward()
        optimizer.step()

        # log data
        num_data += target_y.shape[0]
        epoch_loss += loss.item()
        epoch_acc += (predict_y.argmax(axis=1) == target_y).sum().item()

    mean_loss = epoch_loss / num_data
    mean_acc = epoch_acc / num_data
    return mean_loss, mean_acc


def _test(network, criterion, testLoader, device, need_output_y=False):
    network.eval()
    if need_output_y:
        predict_y_list = []
    epoch_loss = 0
    epoch_acc = 0
    num_data = 0
    for input_x, target_y in tqdm(testLoader):
        input_x, target_y = input_x.to(device), target_y.to(device)
        predict_y = network(input_x)
        loss = criterion(predict_y, target_y)

        # log data
        if need_output_y:
            predict_y_list.append(predict_y.detach().cpu().numpy())
        num_data += target_y.shape[0]
        epoch_loss += loss.item()
        epoch_acc += (predict_y.argmax(axis=1) == target_y).sum().item()

    mean_loss = epoch_loss / num_data
    mean_acc = epoch_acc / num_data
    if need_output_y:
        return mean_loss, mean_acc, predict_y_list
    else:
        return mean_loss, mean_acc


class Recorder():
    def __init__(self, config):
        self.config = config
        self.df_train = pd.DataFrame(columns=["epoch", "loss", "acc"])
        self.df_test = pd.DataFrame(columns=["epoch", "loss", "acc"])

    def __call__(self, epoch, train_out=None, test_out=None):
        if train_out is not None:
            temp_df = pd.DataFrame([[epoch, train_out[0], train_out[1]]], columns=[
                                   "epoch", "loss", "acc"])
            self.df_train = self.df_train.append(temp_df, ignore_index=True)
            self.df_train.to_csv(os.path.join(
                self.config["save_dir"], "train.csv"))
        if test_out is not None:
            temp_df = pd.DataFrame([[epoch, test_out[0], test_out[1]]], columns=[
                                   "epoch", "loss", "acc"])
            self.df_test = self.df_test.append(temp_df, ignore_index=True)
            self.df_test.to_csv(os.path.join(
                self.config["save_dir"], "test.csv"))


def train(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    trainTransform, testTransform = data.get_transforms(config)
    trainLoader = data.get_dataloader(
        config, train=True, transform=trainTransform)
    testLoader = data.get_dataloader(
        config, train=False, transform=testTransform)

    densenet = model.get_model(config).to(config["device"])
    opt, scheduler = optimizer.get_scheduled_optimizer(config, densenet)
    criterion = CrossEntropyLoss()
    recorder = Recorder(config)

    max_epoch = config["max_epoch"]
    for epoch in range(max_epoch):
        print("epoch:{:0>3}".format(epoch))
        train_out = _train(densenet, criterion, trainLoader,
                           config["device"], opt)
        recorder(epoch, train_out=train_out)
        test_out = _test(densenet, criterion, testLoader,
                         config["device"], need_output_y=False)
        recorder(epoch, test_out=test_out)
        scheduler.step()
        torch.save(densenet, os.path.join(config["save_dir"], "latest.pth"))
    print("train finished")
    return None


def test(config, testLoader=None):
    # check if there's weight in work/* (* means the name of setting) before run test.
    if testLoader is None:
        trainTransform, testTransform = data.get_transforms(config)
        testLoader = data.get_dataloader(
            config, train=False, transform=testTransform)

    densenet = model.get_model(config)
    densenet.load_state_dict(os.path.join(config["save_dir"], "latest.pth"))
    criterion = CrossEntropyLoss()
    test_out = _test(densenet, criterion, testLoader, need_output_y=True)
    print("test finished")
    return test_out


def main():
    config = get_config()
    if config["mode"] == "train":
        train(config)
    elif config["mode"] == "test":
        test(config)
    else:
        raise Exception("unknown mode")


if __name__ == "__main__":
    main()
