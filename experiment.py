import torch
import wandb

import jiant.proj.main.tokenize_and_cache as tokenize_and_cache
import jiant.proj.main.export_model as export_model
import jiant.proj.main.scripts.configurator as configurator
import jiant.proj.main.components.container_setup as container_setup
import jiant.shared.caching as caching
import jiant.proj.main.modeling.model_setup as jiant_model_setup
import os

from typing import Union, List
from copy import deepcopy
from loaders import Compose, RandomRotate, RandomHorizontallyFlip, global_transformer
from loaders import MNIST, CITYSCAPES, CIFAR10Loader
from models import MultiDec, MultiLeNetDec, MultiLeNetEnc, SegmentationDecoder, get_segmentation_encoder, ResNet18
from utils import cross_entropy2d, l1_loss_instance, l1_loss_depth, partialclass, set_seed, reset_model


def set_task(DATASET: str,
             BATCH_SIZE: int,
             path: str,
             N_WORKERS: int) -> Union[Dataloader, Dataloader, List, List, List]:
    """
    Setting task parameters
    Args:
        DATASET: Dataset name
        BATCH_SIZE: training batch size
        path: path to dataset folder
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
        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=BATCH_SIZE, shuffle=True,
                                                   num_workers=N_WORKERS)

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
        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=BATCH_SIZE, shuffle=True,
                                                   num_workers=N_WORKERS)

        val_dst = CITYSCAPES(root=path, split=['val'], img_size=(img_rows, img_cols))
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=BATCH_SIZE, num_workers=N_WORKERS)

        list_of_encoders = [get_segmentation_encoder]
        list_of_decoders = [partialclass(SegmentationDecoder, num_class=19, task_type="C"),
                            partialclass(SegmentationDecoder, num_class=2, task_type="R"),
                            partialclass(SegmentationDecoder, num_class=1, task_type="R")]
        criterions = [cross_entropy2d, l1_loss_instance, l1_loss_depth]
    
    elif DATASET == 'NLP':

        export_model.export_model(
            hf_pretrained_model_name_or_path="bert-base-uncased",
            output_base_path="./models/bert-base-uncased",
        )

        for task_name in ["rte", "stsb", "commonsenseqa"]:
            tokenize_and_cache.main(tokenize_and_cache.RunConfiguration(
                task_config_path=f"./tasks/configs/{task_name}_config.json",
                hf_pretrained_model_name_or_path="bert-base-uncased",
                output_dir=f"./cache/{task_name}",
                phases=["train", "val"],
            ))

        jiant_run_config = configurator.SimpleAPIMultiTaskConfigurator(
            task_config_base_path="./tasks/configs",
            task_cache_base_path="./cache",
            train_task_name_list=["rte", "stsb", "commonsenseqa"],
            val_task_name_list=["rte", "stsb", "commonsenseqa"],
            train_batch_size=4,
            eval_batch_size=8,
            epochs=0.5,
            num_gpus=1,
        ).create_config()

        jiant_task_container = container_setup.create_jiant_task_container_from_dict(jiant_run_config)

        jiant_model = jiant_model_setup.setup_jiant_model(
            hf_pretrained_model_name_or_path="bert-base-uncased",
            model_config_path="./models/bert-base-uncased/model/config.json",
            task_dict=jiant_task_container.task_dict,
            taskmodels_config=jiant_task_container.taskmodels_config,
        )

        train_cache = jiant_task_container.task_cache_dict['stsb']["train"]
        val_cache = jiant_task_container.task_cache_dict['stsb']["val"]

        train_dataloader = get_train_dataloader_from_cache(train_cache, task, 4)
        val_dataloader = get_eval_dataloader_from_cache(val_cache, task, 4)

        list_of_encoders = [jiant_model.encoder]
        decoder1 = deepcopy(jiant_model.taskmodels_dict['stsb'].head)
        reset(decoder1)
        decoder2 = deepcopy(decoder1)
        reset(decoder2)
        decoder3 = deepcopy(decoder2)
        reset(decoder3)

        list_of_decoders = [lambda: decoder1, lambda: decoder2, lambda: decoder3]
        criterions = [torch.nn.MSELoss(), torch.nn.MSELoss(), torch.nn.MSELoss()]

    return train_loader, val_loader, criterions, list_of_encoders, list_of_decoders


def init_wandb_log(method: str,
                   i_seed: int,
                   group: str,
                   params: dict) -> None:
    """
    Initializing parameters for wandb logging
    Args:
        method: method for finding optimal direction
        i_seed: seed of experiment
        group: name of experiment group for logging
        params: dictionary with all parameters
    """
    WANDB_NAME, N_EPOCHS, N_DROP_LR = params["WANDB_NAME"], params["N_EPOCHS"], params["N_DROP_LR"]
    DROP_LR_FACTOR, BATCH_SIZE, BETA = params["DROP_LR_FACTOR"], params["BATCH_SIZE"], params["BETA"]
    DATASET, LEARNING_RATE_IN, GRADIENT = params["DATASET"], params["LEARNING_RATE_IN"], params["GRADIENT"]
    entity = params["entity"]
    wandb.init(entity=entity, project=WANDB_NAME, name=method, group=group)
    wandb.config.n_epochs = N_EPOCHS
    wandb.config.n_drop_lr = N_DROP_LR
    wandb.config.drop_lr_factor = DROP_LR_FACTOR
    wandb.config.batch_size = BATCH_SIZE
    wandb.config.initial_lr = LEARNING_RATE_IN
    wandb.config.beta = BETA
    wandb.config.grad = GRADIENT
    wandb.config.dataset = DATASET
    wandb.config.seed = i_seed
