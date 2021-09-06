import functools
import numpy as np
import os
import random
import torch
import torch.nn.functional as F

from typing import List, Tuple, Callable


def set_seed(seed: int) -> None:
    """
    Set reproducibility

    Args:
        seed: seed for experiment

    Returns:

    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def backtracking(losses: List[torch.Tensor],
                 images: torch.Tensor,
                 encoders: List[nn.Module],
                 decoders: List[nn.Module],
                 criterions: List[Callable],
                 new_grads: Tuple[torch.Tensor],
                 enc_output: List[torch.Tensor],
                 trues: torch.Tensor,
                 t: int,
                 GRADIENT: str,
                 BETA: float,
                 total_norms1: float) -> float:
    """
    Implement backtracking procedure

    Args:
        losses: seed for experiment
        images: batch of images
        encoders: encoder models
        decoders: decoder models
        criterions: loss functions
        new_grads: modified gradient direction
        enc_output: output of encoder models
        trues: true labels
        t: int,
        GRADIENT: type of gradient ('dz' or 'dp')
        BETA: parameter of Armijo rule
        total_norms1: norm of decoder parameters,
    Returns:
        Optimal step-size
    """
    with torch.no_grad():
        if GRADIENT == "dz":
            new_grads = new_grads[0]
            enc_output = [enc_output[0] - t * new_grads]
        else:
            for i_par, parameter in enumerate(encoders["enc"].parameters()):
                if parameter.grad is not None:
                    parameter.data -= t * new_grads[i_par].data
            enc_output = apply_encoders(images, encoders)

        outputs, _ = apply_decoders(decoders, enc_output, GRADIENT, train=False)

        crit = np.array(calculate_losses(criterions, outputs, trues)) - losses + t * BETA * total_norms1

        while any(crit > 0) and (t > 1e-10):
            t /= 2.0

            if GRADIENT == "dz":
                enc_output = [enc_output[0] + t * new_grads]
            else:
                for i_par, parameter in enumerate(encoders["enc"].parameters()):
                    if parameter.grad is not None:
                        parameter.data += t * new_grads[i_par].data
                enc_output = apply_encoders(images, encoders)

            outputs, _ = apply_decoders(decoders, enc_output, GRADIENT, train=False)

            for d in decoders:
                for parameter in decoders[d].parameters():
                    if parameter.grad is not None:
                        parameter.data = parameter.data + t * parameter.grad.data

            crit = np.array(calculate_losses(criterions, outputs, trues)) - losses + t * BETA * total_norms1
    return t


def step_norm(decoders: List[nn.Module],
              LEARNING_RATE: float) -> np.array:
    """
    Do gradient step and calculate norm of gradient
    Args:
        decoders: decoder models
        LEARNING_RATE: step-size

    Returns:
        Norm of decoder model
    """
    total_norms = []
    for d in decoders:
        total_norm = 0.0
        for parameter in decoders[d].parameters():
            if parameter.grad is not None:
                param_norm = parameter.grad.data.norm(2)
                parameter.data = parameter.data - LEARNING_RATE * parameter.grad.data
                total_norm += param_norm.item() ** 2
        total_norms.append(total_norm)
    return np.array(total_norms)


def calculate_product(grads: Tuple[torch.Tensor],
                      new_grad: Tuple[torch.Tensor]):
    """
    Calculating scalar product between task gradients and minimizing direction
    Args:
        grads: task gradients
        new_grad: minimizing direction

    Returns:
        Scalar product between direction
    """
    size = len(grads)
    scalar = new_grad[0].new_zeros(size=(size,))
    for i in range(size):
        for j in range(len(new_grad)):
            scalar[i] += (grads[i][j] * new_grad[j]).sum()

    return scalar


def step_zero(decoders: List[nn.Module],
              encoders: List[nn.Module],
              step_size: float,
              GRADIENT: str,
              BACKTRACKING: bool,
              grads: Tuple[torch.Tensor]) -> None:
    """
    encoder gradient update and gradient zeroing
    Args:
        decoders: decoder models
        encoders: encoder models
        step_size: step-size for update
        GRADIENT: type of gradient ('dz' or 'dp')
        BACKTRACKING: Set True if you use backtracking
        grads: gradients

    Returns:
        None
    """
    if GRADIENT == "dz":
        for i_par, parameter in enumerate(encoders['enc'].parameters()):
            if parameter.grad is not None:
                parameter.data = parameter.data - step_size * parameter.grad.data
                parameter.grad.data.zero_()

    elif not BACKTRACKING:
        for i_par, parameter in enumerate(encoders['enc'].parameters()):
            if parameter.grad is not None:
                parameter.data = parameter.data - step_size * grads[i_par].data
                parameter.grad.data.zero_()

    for d in decoders:
        for parameter in decoders[d].parameters():
            if parameter.grad is not None:
                parameter.grad.data.zero_()

    for e in encoders:
        for parameter in encoders[e].parameters():
            if parameter.grad is not None:
                parameter.grad.data.zero_()


def calculate_losses(criterions: List[Callable],
                     outputs: List[torch.Tensor],
                     trues: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Calculating task losses
    Args:
        criterions: loss functions
        outputs: output of models
        trues: true labels

    Returns:
        List of tensors
    """
    losses = []
    for i, criterion in enumerate(criterions):
        loss = criterion(outputs[i], trues[i])
        losses.append(loss)
    return losses


def calculate_predictions(outputs: List[torch.Tensor],
                          trues: List[torch.Tensor],
                          n: int,
                          acc: float):
    """
    Calculating task predictions
    Args:
        outputs: output of models
        trues: true labels
        n: number of elements
        acc: current accuracy

    Returns:

    """
    for i, _ in enumerate(outputs):
        predictions = outputs[i].data.max(1, keepdim=True)[1]
        acc[i] += predictions.eq(trues[i].data.view_as(predictions)).cpu().sum()
        n += predictions.shape[0] if i == 0 else 0

    return n, acc


def binary_acc(outputs, trues, n, acc):
    for i, y_pred in enumerate(outputs):
        predictions = torch.round(torch.sigmoid(y_pred))

        acc[i] += (predictions == trues[i]).cpu().sum()
        n += predictions.shape[0] if i == 0 else 0

    return n, acc


def cross_entropy2d(input, target, weight=None, val=False):
    if val:
        size_average = False
    else:
        size_average = True

    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, ignore_index=250,
                      weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


def l1_loss_depth(input, target, val=False):
    if val:
        size_average = False
    else:
        size_average = True
    mask = target > 0
    if mask.data.sum() < 1:
        # no instance pixel
        return None

    lss = F.l1_loss(input[mask], target[mask], size_average=False)
    if size_average:
        lss = lss / mask.data.sum()
    return lss


def l1_loss_instance(input, target, val=False):
    if val:
        size_average = False
    else:
        size_average = True
    mask = target != 250
    if mask.data.sum() < 1:
        # no instance pixel
        return None

    lss = F.l1_loss(input[mask], target[mask], size_average=False)
    if size_average:
        lss = lss / mask.data.sum()
    return lss


def partialclass(cls, *args, **kwds):
    class NewCls(cls):
        __init__ = functools.partialmethod(cls.__init__, *args, **kwds)

    return NewCls

def reset_model(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
