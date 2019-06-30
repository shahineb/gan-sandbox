from torch.utils.data.dataloader import default_collate

"""
Default batch formatting when using pytorch dataloading modules is done as :
[(data_1, target_1), (data_2, target_2), ... , (data_n, target_n)]

where the latter tuples are usually torch.Tensor instances.

The following utilities are meant to process such input and manipulate data in
order to yield the batches in a more training-compliant fashion
"""


def drop_target(batch):
    """Simply performs default collate operation but drops target in the process

    Args:
        batch (list): batch as described above
    """
    data, _ = default_collate(batch)
    return data
