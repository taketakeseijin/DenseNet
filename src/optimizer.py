
from torch import optim

OPTIMS = dict(
    sgd=optim.SGD,
    adam=optim.Adam
)


def get_optimizer(optimizer_name, network):
    return OPTIMS[optimizer_name](network.parameters(), lr=0.1)


def step_scheduler(optimizer, config):
    max_epoch = config["max_epoch"]
    milestones = [int(max_epoch*0.5), int(max_epoch*0.75)]
    return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)


def custom_scheduler(optmizer, config):
    def lr_lambda(epoch): return 0.1  # custom here
    return optim.lr_scheduler.MultiplicativeLR(optmizer, lr_lambda=lr_lambda)


SCHEDULERS = dict(
    no=lambda opt, config: opt,
    step=step_scheduler,
    custom=custom_scheduler,
)


def get_scheduled_optimizer(config, network):
    optimizer = get_optimizer(config["opt"],network)
    return optimizer, SCHEDULERS[config["scheduler"]](optimizer, config)
