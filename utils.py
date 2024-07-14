import torch


def cal_pred_res(prob):
    test_pred = []
    for i, item in enumerate(prob):
        tmp_label = []
        tmp_label.append(1 - item[0])
        tmp_label.append(item[0] - item[1])
        tmp_label.append(item[1] - item[2])
        tmp_label.append(item[2])
        test_pred.append(tmp_label)
    return test_pred


def cal_pred_resN(prob):
    weight = prob.cumprod(dim=-1)
    ground = torch.ones_like(prob)
    prob = ground - prob
    prob = torch.concat([prob, torch.ones((prob.shape[0], 1), device="cuda")], dim=-1)
    weight = torch.concat([torch.ones((weight.shape[0], 1), device="cuda"), weight], dim=-1)
    return (prob * weight).flip(dims=(-1,))


def weighted_mse_loss(pred, gt, weight):
    return torch.sum(weight * (pred - gt) ** 2)


def cal_pred_resT(prob):
    return prob.flip(dims=(-1,)).diff(dim=1, prepend=torch.zeros(prob.shape[0], 1, device="cuda"),
                                      append=torch.ones(prob.shape[0], 1, device="cuda")).flip(dims=(-1,))



def chaos_loss_fun(h, z, I0=0.65):
    out = torch.sigmoid(h / 10)
    log1, log2 = torch.log(out), torch.log(1 - out)
    return -z * (I0 * log1 + (1 - I0) * log2)


def chaos(hid, z, chaos_loss_f, I0=0.65):
    chaosloss = 0
    if type(hid) == torch.Tensor:
        hid = torch.where(hid < 9.9, hid.float(), torch.tensor(9.9).to(hid.device))
        hid = torch.where(hid > -10, hid, torch.tensor(-10.0).to(hid.device))
        chaosloss = chaos_loss_f(hid, z).sum() / hid.numel()
    else:
        for i in range(len(hid) - 1):
            hid_thresold = torch.where(hid[i + 1] < 9.9, hid[i + 1], torch.tensor(9.9).to(hid[i + 1].device))
            hid_thresold = torch.where(hid_thresold > -10.0, hid_thresold, torch.tensor(-10.0).to(hid[i + 1].device))
            chaosloss += chaos_loss_f(hid_thresold, z).sum() / hid_thresold.numel()

    return chaosloss
