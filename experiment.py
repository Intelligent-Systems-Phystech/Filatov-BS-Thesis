import torch
import numpy as np
import random
import os
import wandb

from models import MultiDec, MultiLeNetDec, MultiLeNetEnc, SegmentationDecoder, get_segmentation_encoder, ResNet18
from utils import cross_entropy2d, l1_loss_instance, l1_loss_depth, partialclass, set_seed
from loaders import Compose, RandomRotate, RandomHorizontallyFlip, global_transformer
from loaders import MNIST, CITYSCAPES, CIFAR10Loader

def set_task(DATASET, BATCH_SIZE, path, N_WORKERS):
    """
    Setting task parameters
    Args:
        DATASET: Dataset name
        BATCH_SIZE: training batch size
        path: path to dataset
        N_WORKERS: num workers

    Returns:
    train_loader - loader for training set
    val_loader - loader for validation set
    criterions - loss functions
    list_of_encoders - encoder models
    list_of_decoders - decoder models
    """
    set_seed(999)
    if DATASET == "CIFAR-10":
        train_dst = CIFAR10Loader(root=path, train=True)
        train_loader = train_dst.get_loader(batch_size=BATCH_SIZE, shuffle=True)

        val_dst = CIFAR10Loader(root=path, train=False)
        val_loader = val_dst.get_loader()

        list_of_encoders = [ResNet18]
        list_of_decoders = [MultiDec] * 10
        criterions = [torch.nn.BCEWithLogitsLoss()] * 10

    elif DATASET == "MNIST":
        train_dst = MNIST(root=path, train=True, download=True, transform=global_transformer(), multi=True)
        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)

        val_dst = MNIST(root=path, train=False, download=True, transform=global_transformer(), multi=True)
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=BATCH_SIZE, num_workers=N_WORKERS)

        list_of_encoders = [MultiLeNetEnc]
        list_of_decoders = [MultiLeNetDec] * 2
        criterions = [torch.nn.NLLLoss()] * 2

    elif DATASET == "Cityscapes":
        cityscapes_augmentations = Compose([RandomRotate(10), RandomHorizontallyFlip()])
        img_rows = 256
        img_cols = 512

        train_dst = CITYSCAPES(root=path, is_transform=True, split=['train'],
                               img_size=(img_rows, img_cols), augmentations=cityscapes_augmentations)
        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_WORKERS)

        val_dst = CITYSCAPES(root=path, split=['val'], img_size=(img_rows, img_cols))
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=BATCH_SIZE, num_workers=N_WORKERS)

        list_of_encoders = [get_segmentation_encoder]
        list_of_decoders = [partialclass(SegmentationDecoder, num_class=19, task_type="C"),
                            partialclass(SegmentationDecoder, num_class=2, task_type="R"),
                            partialclass(SegmentationDecoder, num_class=1, task_type="R")]
        criterions = [cross_entropy2d, l1_loss_instance, l1_loss_depth]

    return train_loader, val_loader, criterions, list_of_encoders, list_of_decoders


def init_wandb_log(method, i_seed, group, params):
    """
    Initializing parameters for wandb logging
    Args:
        method: method for finding optimal direction
        i_seed: seed of experiment
        group: experiment group
        params: dictionary with all parameters
    """
    WANDB_NAME, N_EPOCHS, N_DROP_LR = params["WANDB_NAME"], params["N_EPOCHS"], params["N_DROP_LR"]
    DROP_LR_FACTOR, BATCH_SIZE, BETA = params["DROP_LR_FACTOR"], params["BATCH_SIZE"], params["BETA"]
    DATASET, LEARNING_RATE_IN, GRADIENT = params["DATASET"], params["LEARNING_RATE_IN"], params["GRADIENT"]
    entity = params["entity"]
    wandb.init(entity=entity, project=WANDB_NAME, name=method, group=group)
    wandb.config.n_epochs       = N_EPOCHS
    wandb.config.n_drop_lr      = N_DROP_LR
    wandb.config.drop_lr_factor = DROP_LR_FACTOR
    wandb.config.batch_size     = BATCH_SIZE
    wandb.config.initial_lr     = LEARNING_RATE_IN
    wandb.config.beta           = BETA
    wandb.config.grad           = GRADIENT
    wandb.config.dataset        = DATASET
    wandb.config.seed           = i_seed
