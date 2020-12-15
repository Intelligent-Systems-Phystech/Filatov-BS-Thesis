import torch
import numpy as np
import wandb
import json
import os

from argparse import ArgumentParser
from tqdm import tqdm
from experiment import set_task, init_wandb_log
from utils import get_models, apply_decoders, apply_encoders, calculate_losses, calculate_predictions, binary_acc
from utils import step_norm, step_zero, backtracking, set_seed, calculate_product
from methods import change_gradient

def train(params):
    DATASET = params["DATASET"]
    BATCH_SIZE = params["BATCH_SIZE"]
    N_experiments = params["N_experiments"]
    group = params["group"]
    N_DROP_LR = params["N_DROP_LR"]
    DROP_LR_FACTOR = params["DROP_LR_FACTOR"]
    N_EPOCHS = params["N_EPOCHS"]
    path = params["path"]
    DEVICE = torch.device(params["device"])
    N_WORKERS = params["n_workers"]

    seeds = range(999, 999-N_experiments, -1)
    train_loader, val_loader, criterions, list_of_encoders, list_of_decoders = set_task(DATASET, BATCH_SIZE, path, N_WORKERS)
    list_of_methods = params['methods']

    for i_exp, i_seed in tqdm(zip(range(N_experiments), seeds)):
        for method, GRADIENT, LEARNING_RATE_IN, BETA in list_of_methods:
            BACKTRACKING = BETA > 0.0
            params["BACKTRACKING"] = BACKTRACKING
            params["BETA"] = BETA
            params["LEARNING_RATE_IN"] = LEARNING_RATE_IN
            params["GRADIENT"] = GRADIENT
            params["LEARNING_RATE"] = LEARNING_RATE_IN

            init_wandb_log(method, i_seed, group, params)
            # Reproducibility
            set_seed(i_seed)

            encoders, decoders = get_models(params, list_of_encoders, list_of_decoders, DEVICE)

            n_iter = 0
            LEARNING_RATE = LEARNING_RATE_IN
            print(f'{DATASET} training with {method}.')

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            torch.cuda.synchronize()
            start.record()

            for i_epoch in tqdm(range(N_EPOCHS)):
                if (i_epoch > 0) and (i_epoch+1) % N_DROP_LR == 0:
                    LEARNING_RATE_IN *= DROP_LR_FACTOR

                start1 = torch.cuda.Event(enable_timing=True)
                end1 = torch.cuda.Event(enable_timing=True)

                torch.cuda.synchronize()
                start1.record()

                n_train, n_test = 0, 0
                if DATASET == "CIFAR-10":
                    train_acc = [0.0] * 10
                    test_acc = [0.0] * 10

                elif DATASET == "MNIST":
                    train_acc = [0.0] * 2
                    test_acc = [0.0] * 2

                for m in decoders:
                    decoders[m].train()

                for m in encoders:
                    encoders[m].train()

                for batch in train_loader:
                    n_iter += 1
                    LEARNING_RATE = LEARNING_RATE_IN
                    images, trues = batch
                    images = images.to(DEVICE)

                    if DATASET == "CIFAR-10":
                        trues = trues.T.unsqueeze(2).to(DEVICE)
                    elif DATASET == "MNIST":
                        for i in range(len(trues)):
                            trues[i] = trues[i].to(DEVICE)
                    elif DATASET == "Cityscapes":
                        for i in range(len(trues)):
                            trues[i] = trues[i].to(DEVICE)

                    enc_output = apply_encoders(images, encoders)
                    if GRADIENT == "dz":
                        outputs, t_outputs = apply_decoders(decoders, enc_output, GRADIENT, train=True)
                    else:
                        outputs, _ = apply_decoders(decoders, enc_output, GRADIENT, train=True)

                    # Task-specific losses
                    losses = calculate_losses(criterions, outputs, trues)
                    if DATASET == "CIFAR-10":
                        n_train, train_acc = binary_acc(outputs, trues, n_train, train_acc)
                    elif DATASET == "MNIST":
                        n_train, train_acc = calculate_predictions(outputs, trues, n_train, train_acc)
                    elif DATASET == "Cityscapes":
                        pass

                    grads = []
                    if GRADIENT == "dz":
                        for i, loss in enumerate(losses):
                            loss.backward(retain_graph=True)
                            if len(t_outputs) == 1:
                                gs = t_outputs[0].grad.clone()
                                t_outputs[0].grad.zero_()
                            else:
                                gs = t_outputs[i].grad.clone()
                                t_outputs[i].grad.zero_()
                            grads.append(tuple(gs.unsqueeze(0)))

                    else:
                        for loss in losses:
                            loss.backward(retain_graph=True)
                            gs = []
                            for parameter in encoders['enc'].parameters():
                                if parameter.grad is not None:
                                    gs.append(parameter.grad.data.clone())
                                    parameter.grad.data.zero_()
                            grads.append(tuple(gs))

                    loss_vector = np.fromiter(map(lambda x: x.item(), losses), dtype=float)
                    # Updating shared weights

                    new_grads = change_gradient(method, grads)
                    if GRADIENT == "dz":
                        enc_output[0].backward(new_grads[0])

                    total_norms1 = step_norm(decoders, LEARNING_RATE)
                    if BACKTRACKING:
                        total_norms2 = calculate_product(grads, new_grads)
                        LEARNING_RATE = backtracking(loss_vector, images, encoders, decoders, criterions,
                                                    new_grads, enc_output, trues, LEARNING_RATE, GRADIENT, BETA,
                                                    total_norms1, total_norms2)

                    step_zero(decoders, encoders, LEARNING_RATE, GRADIENT, BACKTRACKING, new_grads)

                    for i, _ in enumerate(losses):
                        wandb.log({f'Train loss {i}': losses[i].data})

                    wandb.log({'Epoch': i_epoch + 1,
                               'Iterations': n_iter,
                               'Step size': LEARNING_RATE})

                train_acc = list(map(lambda x: float(x) / n_train, train_acc))
                if DATASET != "Cityscapes":
                    for i, _ in enumerate(train_acc):
                        wandb.log({f'Train error {i}': 1 - train_acc[i]})

                wandb.log({'Epoch': i_epoch + 1,
                           'Iterations': n_iter})

                for m in encoders:
                    encoders[m].eval()

                for m in decoders:
                    decoders[m].eval()

                for batch in val_loader:
                    # Here can be problems with single task setup
                    images, trues = batch
                    images = images.to(DEVICE)

                    if DATASET == "CIFAR-10":
                        trues = trues.T.unsqueeze(2).to(DEVICE)
                    elif DATASET == "MNIST":
                        for i in range(len(trues)):
                            trues[i] = trues[i].to(DEVICE)
                    elif DATASET == "Cityscapes":
                        for i in range(len(trues)):
                            trues[i] = trues[i].to(DEVICE)

                    enc_output = apply_encoders(images, encoders)
                    outputs, _ = apply_decoders(decoders, enc_output, GRADIENT, train=False)
                    # Task-specific losses

                    losses = calculate_losses(criterions, outputs, trues)
                    if DATASET == "CIFAR-10":
                        n_test, test_acc = binary_acc(outputs, trues, n_test, test_acc)
                    elif DATASET == "MNIST":
                        n_test, test_acc = calculate_predictions(outputs, trues, n_test, test_acc)
                    elif DATASET == "Cityscapes":
                        pass

                test_acc = list(map(lambda x: float(x) / n_test, test_acc))

                end1.record()
                torch.cuda.synchronize()

                comp_time1 = start1.elapsed_time(end1) / 1000

                wandb.log({"Epoch time": comp_time1})

                for i, acc in enumerate(losses):
                    wandb.log({f'Test loss. {i}': losses[i].data})
                    if DATASET != "Cityscapes":
                        wandb.log({f'Test error. {i}': 1 - test_acc[i]})

                wandb.log({
                    'Epoch': i_epoch + 1,
                    'Learning rate': LEARNING_RATE,
                    'Iterations': n_iter})

            end.record()
            torch.cuda.synchronize()

            comp_time = start.elapsed_time(end) / 1000

            wandb.log({"Full time": comp_time})
            if DATASET != "Cityscapes":
                for i, acc in enumerate(test_acc):
                    wandb.log({f"Final test error. {i}": 1 - test_acc[i]})
                    wandb.log({f"Final test accuracy. {i}": test_acc[i]})

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("path", type=str)
    arg_parser.add_argument("-device", type=str, default="cpu")
    arg_parser.add_argument("-n_workers", type=int, default=4)
    arg_parser.add_argument("-logging", default="false", choices=["true", "false"])
    args = arg_parser.parse_args()

    with open(args.path, "r") as json_file:
        params = json.load(json_file)
        params["device"] = args.device
        params["n_workers"] = args.n_workers
        params["single_task"] = False

    if args.logging == "false":
        os.environ["WANDB_MODE"] = 'disabled'
        os.environ["WANDB_SILENT"] = "true"
    train(params)
