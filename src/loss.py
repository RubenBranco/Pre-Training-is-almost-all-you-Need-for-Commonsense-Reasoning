import torch


def ssm_loss(ssm, y, threshold):
    losses = torch.zeros(ssm.shape[0]).to(ssm.device)

    for i in range(losses.shape[0]):
        ssm_y = ssm[i, y[i]]
        ssm_diff_y = ssm[i, torch.arange(ssm.shape[1]).to(ssm.device) != y[i]]
        loss = (
            ssm_diff_y
            - ssm_y.repeat(ssm_diff_y.shape[0])
            + torch.Tensor([threshold]).to(ssm.device).repeat(ssm_diff_y.shape[0])
        )
        loss = (
            torch.max(loss, torch.zeros(ssm_diff_y.shape[0]).to(ssm.device)).sum()
            / ssm_diff_y.shape[0]
        )
        losses[i] = loss

    return losses.mean()
