import torch


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
    positives = torch.nonzero(predicted > thresh)
    true_positives = (predicted[positives].float() == groundtruth[positives]).sum().float()
    precision = true_positives / len(positives)
    return precision.item()


def recall(predicted, groundtruth, thresh=0.5):
    """Recall on single label classification
    Args:
        predicted (torch.Tensor): batch of probability distributions on classes
        groundtruth (torch.Tensor): batch of probability distributions on classes
    """
    predicted = predicted > thresh
    positives = torch.nonzero(groundtruth == 1)
    true_positives = (predicted[positives].float() == groundtruth[positives]).sum().float()
    recall = true_positives / len(positives)
    return recall.item()
