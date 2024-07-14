import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from wfdb import processing
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
def cal_pred_resT(prob):
    return prob.flip(dims=(-1,)).diff(dim=1,prepend=torch.zeros(prob.shape[0], 1, device="cpu"),append=torch.ones(prob.shape[0],1,device="cpu")).flip(dims=(-1,))
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

value = []
weights = []
def forward_hook(module, args, output):
    # value.append(args)
    value.append(output)


def hook_grad(module, grad_input, grad_output):
    # weights.append(grad_input)
    weights.append(grad_output)


# cam_test(model, show_case[i], (show_res[i], target), 'stage_list.4.block_list.0.conv1.conv', True)
def cam_test(
        model, # Model
        data,  # ECG data signal my shape is torch.Size([1, 187])
        clas,  # a tuple with (predict, ground_truth)
        layer, # target layer to compute the graph like 'stage_list.4.block_list.0.conv1.conv'
        label,
        show_overlay=False,
        save=False):# if show the heatmap of signal

    #####################################################
    # register hooks

    data = data.reshape(1,50, 1, 721)
    output_len = data.shape[-1]
    target_layer = model.get_submodule(layer)
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(hook_grad)
    # clear list
    weights.clear()
    model.zero_grad()
    # forward
    value.clear()
    out,_,_ = model(torch.tensor(data))
    out = cal_pred_resT(out).mean(0)
    loss = out[int(clas[0])]
    loss.backward(retain_graph=True)
    weight = weights[0][0].mean(-1)
    missing_dims = value[0].ndim - weight.ndim
    weight = weight[(...,) + (None,) * missing_dims]
    cam = value[0] * weight
    p_acti_map = F.relu(cam.sum(1))

    # gt class actimap
    weights.clear()
    model.zero_grad()
    loss = out[int(clas[1])]
    loss.backward()
    weight = weights[0][0].mean(-1)[(...,) + (None,) * missing_dims]
    cam = value[0] * weight
    g_acti_map = F.relu(cam.sum(1))

    forward_handle.remove()
    backward_handle.remove()

    #######################################################
    # build colored ECG
    #######################################################
    if show_overlay:
        p_acti_map = p_acti_map.detach().numpy()
        g_acti_map = g_acti_map.detach().numpy()
        p_new_acti = processing.resample_sig(p_acti_map.mean(1).flatten(), p_acti_map.shape[-1], output_len)[0]
        g_new_acti = processing.resample_sig(g_acti_map.mean(1).flatten(), g_acti_map.shape[-1], output_len)[0]

        p_new_acti = pd.Series(p_new_acti)
        g_new_acti = pd.Series(g_new_acti)
        p_new_acti = p_new_acti.interpolate()
        g_new_acti = g_new_acti.interpolate()
        # 方便画图用的定义
        x = np.arange(0, output_len)
        y = data[0][0][0].flatten()

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)

        dydx = p_new_acti
        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(dydx.min(), dydx.max())
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        # Set the values used for colormapping
        lc.set_array(dydx)
        lc.set_linewidth(2)
        line = axs[0].add_collection(lc)
        fig.colorbar(line, ax=axs[0])

        dydx = g_new_acti
        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(dydx.min(), dydx.max())
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        # Set the values used for colormapping
        lc.set_array(dydx)
        lc.set_linewidth(2)
        line = axs[1].add_collection(lc)
        fig.colorbar(line, ax=axs[1])

        axs[0].set_xlim(x.min(), x.max())
        axs[0].set_ylim(y.min()-0.3, y.max()+0.3)
        axs[0].set_title('predict class:' + clas[0].__str__())
        axs[1].set_xlim(x.min(), x.max())
        axs[1].set_ylim(y.min() - 0.3, y.max() + 0.3)
        axs[1].set_title('ground truth class:' + clas[1].__str__())
        if(save == True):
            plt.savefig(f'CAM/{label}.pdf')
            plt.close(fig)
        else:
            plt.show()
    ################################################################
    return p_acti_map

def PO_CAM(model, # Model
        data,  # ECG data signal my shape is torch.Size([1, 187])
        res,
        tar,  # a tuple with (predict, ground_truth)
        layer, # target layer to compute the graph like 'stage_list.4.block_list.0.conv1.conv'
        label,
        show_overlay=False,
        save=False):
    #####################################################
    # register hooks
    data = torch.tensor(data)
    output_len = data.shape[-1]
    target_layer = model.get_submodule(layer)
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(hook_grad)
    # clear list
    weights.clear()
    model.zero_grad()
    # forward
    value.clear()
    # data = (data - data.min())*6 / (data.max()-data.min())
    out,_,_ = model(data.reshape(721).tile(50, 1, 1, 1))
    out = cal_pred_resT(out).mean(0)
    # clas = out.argmax(-1)
    clas = res
    loss = out[clas]
    loss.backward(retain_graph=True)
    weight = weights[0][0].mean(-1)
    missing_dims = value[0].ndim - weight.ndim
    weight = weight[(...,) + (None,) * missing_dims]
    cam = value[0] * weight
    p_acti_map = F.relu(cam.sum(2))

    forward_handle.remove()
    backward_handle.remove()

    #######################################################
    # build colored ECG
    #######################################################

    p_acti_map = p_acti_map.detach().cpu().numpy()
    p_new_acti = processing.resample_sig(p_acti_map.mean(1).mean(0).flatten(), p_acti_map.shape[-1], output_len)[0]

    p_new_acti = pd.Series(p_new_acti)
    p_new_acti = p_new_acti.interpolate()
    p_new_acti = (p_new_acti-p_new_acti.min())/(p_new_acti.max()-p_new_acti.min())
    # 方便画图用的定义
    x = np.arange(0, output_len)
    y = data.flatten().numpy()

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, axs = plt.subplots(1, 1, figsize=(6, 3), dpi=300)

    dydx = p_new_acti
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(dydx.min(), dydx.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(dydx)
    lc.set_linewidth(2)
    line = axs.add_collection(lc)
    fig.colorbar(line, ax=axs)

    axs.set_xlim(x.min(), x.max())
    axs.set_ylim(y.min() - 0.3, y.max() + 0.3)
    axs.set_title('predict class:G' + clas.item().__str__() + "  ground truth:G"+tar.__str__())
    plt.tight_layout()
    # plt.show()
    if (save == True):
        plt.savefig(f'CAM/{label}.pdf')
        plt.close(fig)
    else:
        plt.show()