import torch

def forrward(input):
    """
    :param input:   [batch_size, feat_num, embedding_size]
    :return:
    """
    square_of_sum = torch.pow(torch.sum(input, dim=1),2)
    sum_of_square = torch.sum(input * input, dim=1)
    res = square_of_sum - sum_of_square
    res = 0.5 * torch.sum(res, dim=2)
    return res