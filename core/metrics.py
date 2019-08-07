import torch
import torch.nn.functional as F


def accuracy(predicted, groundtruth, thresh=0.5):
    """Accuracy on single label classification
    Args:
        predicted (torch.Tensor): batch of probability distributions on classes
        groundtruth (torch.Tensor): batch of probability distributions on classes
    """
    predicted = predicted > thresh
    correct = (predicted.float() == groundtruth).sum().float()
    score = correct / len(groundtruth)
    return score.item()


def precision(predicted, groundtruth, thresh=0.5):
    """Precision on single label classification
    Args:
        predicted (torch.Tensor): batch of probability distributions on classes
        groundtruth (torch.Tensor): batch of probability distributions on classes
    """
    predicted = predicted > thresh
    positives = torch.nonzero(predicted).flatten()
    true_positives = torch.sum(predicted[positives].float() == groundtruth[positives]).float()
    precision = true_positives / len(positives)
    return precision.item()


def recall(predicted, groundtruth, thresh=0.5):
    """Recall on single label classification
    Args:
        predicted (torch.Tensor): batch of probability distributions on classes
        groundtruth (torch.Tensor): batch of probability distributions on classes
    """
    predicted = predicted > thresh
    positives = torch.nonzero(groundtruth == 1).flatten()
    true_positives = torch.sum(predicted[positives].float() == groundtruth[positives]).float()
    recall = true_positives / len(positives)
    return recall.item()


def inception_score(fake_samples, inception_model, split_size=4):
    """Salimans et al. (2016)
    Args:
        fake_samples (torch.Tensor): batch of fake generated images
        inception_model (nn.Module): inception model
        split_size (int): number of samples to consider for marginal computation
    """
    with torch.no_grad():
        pred = inception_model(fake_samples)
    conditionals = torch.Tensor(pred.split(split_size))
    marginals = conditionals.mean(dim=1, keepdim=True).repeat(1, split_size, 1)
    kl = F.kl_div(conditionals.view_as(pred), marginals.view_as(pred)).mean()
    return torch.exp(kl).item()
