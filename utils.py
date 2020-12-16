import wandb
import torch
import random
import numpy as np
import os
import functools
import torch.nn.functional as F


def set_seed(seed):
    """
    Set reproducibility

    Args:
        seed:

    Returns:

    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_models(params, list_encoders, list_decoders, device):
    """
    Initialize neural network modules
    Args:
        params:
        list_encoders:
        list_decoders:
        device:

    Returns:
    Initialized models
    """
    encoders, decoders = {}, {}
    if params['single_task'] == True:
        for i in range(len(list_encoders)):
            encoders[f'enc_{i}'] = list_encoders[i]()
            encoders[f'enc_{i}'].to(device)
    else:
        encoders['enc'] = list_encoders[0]()
        encoders['enc'].to(device)

    for i, dec in enumerate(list_decoders):
        decoders[i] = list_decoders[i]()
        decoders[i].to(device)

    return encoders, decoders

def apply_encoders(images, encoders):
    """
    Apply encoder to input images
    Args:
        images:
        encoders:

    Returns:

    """
    enc_outputs = []
    for e in encoders:
        Z = encoders[e](images)
        enc_outputs.append(Z)
    return enc_outputs


def apply_decoders(decoders, enc_outputs, GRADIENT, train=True):
    """
    Apply decoder for latent space
    Args:
        decoders:
        enc_outputs:
        GRADIENT:
        train:

    Returns:

    """
    outputs = []
    t_outputs = []
    first_calculation = 0

    for i, d in enumerate(decoders):
        if len(enc_outputs) == 1:
            if GRADIENT == "dz" and train:
                if first_calculation == 0:
                    t = enc_outputs[0].detach().requires_grad_(True)
                    t_outputs.append(t)
                    first_calculation = 1

                Y = decoders[d](t)
            else:
                Y = decoders[d](enc_outputs[0])
        else:
            if GRADIENT == "dz" and train:
                t = enc_outputs[i].detach().requires_grad_(True)
                Y = decoders[d](t)
            else:
                Y = decoders[d](enc_outputs[i])

            t_outputs.append(t)
        outputs.append(Y)

    return outputs, t_outputs


def backtracking(losses, images, encoders, decoders, criterions,
                 new_grads, enc_output, trues, t, GRADIENT, BETA,
                 total_norms1, total_norms2):
    """
    Implement backtracking procedure
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


def step_norm(decoders, LEARNING_RATE):
    """
    Do gradient step and calculate norm of gradient
    Args:
        decoders:
        LEARNING_RATE:

    Returns:

    """
    total_norms = []
    for d in decoders:
        total_norm = 0.0
        for parameter in decoders[d].parameters():
            if parameter.grad is not None:
                param_norm = parameter.grad.data.norm(2)
                parameter.data = parameter.data - LEARNING_RATE*parameter.grad.data
                total_norm += param_norm.item() ** 2
        total_norms.append(total_norm)
    return np.array(total_norms)

def calculate_product(grads, new_grad):
    """
    Calculating scalar product between task gradients and minimizing direction
    Args:
        grads:
        new_grad:

    Returns:

    """
    size = len(grads)
    scalar = new_grad[0].new_zeros(size=(size,))
    for i in range(size):
        for j in range(len(new_grad)):
            scalar[i] += (grads[i][j] * new_grad[j]).sum()

    return scalar

def step_zero(decoders, encoders, step_size, GRADIENT, BACKTRACKING, grads):
    """
    encoder gradient update
    Args:
        decoders:
        encoders:
        step_size:
        GRADIENT:
        BACKTRACKING:
        grads:

    Returns:

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



def calculate_losses(criterions, outputs, trues):
    """
    Calculating task losses
    Args:
        criterions:
        outputs:
        trues:

    Returns:

    """
    losses = []
    for i, criterion in enumerate(criterions):
        loss = criterion(outputs[i], trues[i])
        losses.append(loss)
    return losses


def calculate_predictions(outputs, trues, n, acc):
    """
    Calculating task predictions
    Args:
        outputs:
        trues:
        n:
        acc:

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
