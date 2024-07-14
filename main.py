import argparse
import ast
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import loader
from utils import cal_pred_res, chaos_loss_fun, chaos
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
from spikingjelly.activation_based import neuron, functional, surrogate, layer, monitor
from torch.utils.tensorboard import SummaryWriter
import os
import datetime
from tqdm import tqdm
from COSNN import MyDataset, COSNN


def cross_validation(student, teacher, k, structure, writer, train=None, val=None, ord=True):
    dataset = MyDataset(train, ord)
    dataset_val = MyDataset(val, ord)

    # teacher

    student.to(device)
    teacher.to(device)
    optimizer = optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-3)
    optimizerT = optim.Adam(teacher.parameters(), lr=1e-4, weight_decay=1e-3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    loss_func1 = nn.BCELoss()
    loss_func2 = nn.MSELoss()

    # Train ANN
    step = 0
    for epoch in tqdm(range(n_epoch * 3), desc="epochT"):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, drop_last=False, shuffle=True)

        # train
        teacher.train()

        prog_iter = tqdm(dataloader, desc="Training")
        for batch_idx, batch in enumerate(prog_iter):
            input_x, input_y = tuple(t.to(device) for t in batch)

            predT = torch.sigmoid(teacher(input_x.permute((1, 0, 2, 3))[0]))

            # Net1d
            lossp = 0.5 * (predT.diff().relu() ** 2).sum(-1).mean()
            lossT = loss_func1(predT, input_y)
            loss = lossT + lossp

            optimizerT.zero_grad()
            loss.backward()
            # optimizer.step()
            optimizerT.step()
            writer.add_scalar('{}-fold/trainT'.format(k), lossT.item(), step)
            step += 1

        # save model
        if epoch == (n_epoch - 1):
            model_save_pathT = os.path.join(model_path,
                                            '{}-{}-{}-fold_epoch_{}_params_file.pkl'.format(structure, "teacher", k,
                                                                                            epoch))
            torch.save(teacher, model_save_pathT)

        # val
        # student.eval()
        teacher.eval()
        prog_iter_val = tqdm(dataloader_val, desc="Validation")
        val_pred_prob = []
        val_labels = []
        with (torch.no_grad()):
            for batch_idx, batch in enumerate(prog_iter_val):
                input_x, input_y = tuple(t.to(device) for t in batch)

                predT = torch.sigmoid(teacher(input_x.permute((1, 0, 2, 3))[0]))
                predT = predT.cpu().data.numpy()
                if ord:
                    predT = cal_pred_res(predT)
                val_pred_prob.append(predT)
                val_labels.append(input_y.cpu().data.numpy())

        val_labels = np.concatenate(val_labels)
        val_pred_prob = np.concatenate(val_pred_prob)

        all_pred = np.argmax(val_pred_prob, axis=1)
        if ord:
            all_gt = np.sum(val_labels, axis=1)
        else:
            all_gt = np.argmax(val_labels, axis=1)

        writer.add_scalar('{}-fold/AccT'.format(k), accuracy_score(all_gt, all_pred), epoch)
        writer.add_scalar('{}-fold/valT'.format(k), mean_squared_error(all_pred, all_gt), epoch)

    # Fix
    for param in teacher.parameters():
        param.requires_grad = False

    # Knowledge Distillation
    step = 0
    z = 1
    beta = 0.01
    om = monitor.OutputMonitor(net, neuron.IFNode)
    st = []

    def surface_hook_teacher(module, args, output):
        st.append(output)

    dt = []

    def deep_hook_teacher(module, args, output):
        dt.append(output)

    teacher.get_submodule('basicblock_list.0.conv1.conv').register_forward_hook(surface_hook_teacher)
    teacher.get_submodule('basicblock_list.2.conv1.conv').register_forward_hook(deep_hook_teacher)

    for epoch in tqdm(range(n_epoch), desc="epochS"):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, drop_last=False, shuffle=True)

        # train
        student.train()

        prog_iter = tqdm(dataloader, desc="Training")
        for batch_idx, batch in enumerate(prog_iter):
            input_x, input_y = tuple(t.to(device) for t in batch)

            predT = torch.sigmoid(teacher(input_x.permute((1, 0, 2, 3))[0]))

            predS, c1out, c2out = student(input_x.permute((1, 0, 2, 3)))

            conv_one = nn.Conv1d(kernel_size=1, in_channels=48, out_channels=32).cuda()
            chaos_loss = chaos(torch.concat([om.records[0].mean(0).flatten(),
                                             om.records[1].mean(0).flatten()]), z, chaos_loss_fun)
            z *= 0.9
            lossF = loss_func2(dt[0][:, :, :-1], conv_one(c2out.mean(0))) + loss_func2(st[0], conv_one(c1out.mean(0)))
            lossD = loss_func2(predS, predT)  # BCELoss (teacher, student)
            lossA = loss_func2(predS, input_y)  # MSELoss (student, target)
            lossp = 0.5 * (predS.diff().relu() ** 2).sum(-1).mean()
            loss = 0.1 * lossD + 0.9 * lossA + chaos_loss * lossF + lossp

            om.clear_recorded_data()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            st.clear()
            dt.clear()
            writer.add_scalar('{}-fold/train'.format(k), loss.item(), step)
            step += 1
            functional.reset_net(student)

        scheduler.step(epoch)
        z *= beta
        # save model
        if epoch == (n_epoch - 1):
            om.remove_hooks()
            model_save_pathS = os.path.join(model_path,
                                            '{}-{}-{}-fold_epoch_{}_params_file.pkl'.format(structure, "student", k,
                                                                                            epoch))
            torch.save(student, model_save_pathS)

        # val
        student.eval()
        teacher.eval()
        prog_iter_val = tqdm(dataloader_val, desc="Validation")
        val_pred_prob = []
        val_labels = []
        with (torch.no_grad()):
            for batch_idx, batch in enumerate(prog_iter_val):
                input_x, input_y = tuple(t.to(device) for t in batch)

                predS, _, _ = student(input_x.permute((1, 0, 2, 3)))

                pred = predS.cpu().data.numpy()
                if ord:
                    pred = cal_pred_res(pred)
                val_pred_prob.append(pred)
                val_labels.append(input_y.cpu().data.numpy())

                functional.reset_net(student)

        val_labels = np.concatenate(val_labels)
        val_pred_prob = np.concatenate(val_pred_prob)

        all_pred = np.argmax(val_pred_prob, axis=1)
        if ord:
            all_gt = np.sum(val_labels, axis=1)
        else:
            all_gt = np.argmax(val_labels, axis=1)

        writer.add_scalar('{}-fold/Acc'.format(k), accuracy_score(all_gt, all_pred), epoch)
        writer.add_scalar('{}-fold/val'.format(k), mean_squared_error(all_pred, all_gt), epoch)

    # torch.save(chaotics,'chaotics.pth')
    print(accuracy_score(all_gt, all_pred))
    process = pd.DataFrame(val_pred_prob)
    aucs = []
    for i in range(4):
        roc_auc = roc_auc_score((all_gt == i), process[i])
        aucs.append(roc_auc)

    return all_gt, all_pred, aucs, process


fea = []


def deep_hook_teacher(module, args, output):
    fea.append(output)


def test_res(model, test, fold, snn=True, ord=True, path=None):
    test_set = MyDataset(test, ord=ord)
    dataloader_test = DataLoader(test_set, batch_size=1, drop_last=False, shuffle=False)
    prog_iter_test = tqdm(dataloader_test, desc="Testing")

    model.eval()
    model.to(device)
    diff_x = []
    diff_y = []
    y = [[], [], [], []]
    labels = [[], [], [], []]
    p = []
    y = []
    x = []
    f = []
    # IFMonitor = monitor.OutputMonitor(net, neuron.IFNode)
    # LIFMonitor = monitor.OutputMonitor(net, neuron.LIFNode)

    # model.get_submodule('conv13').register_forward_hook(deep_hook_teacher)
    with torch.no_grad():
        for batch_idx, batch in enumerate(prog_iter_test):
            input_x, input_y = tuple(t.to(device) for t in batch)
            if snn:

                pred, _, feature = model(input_x.permute((1, 0, 2, 3)))

            else:
                for i in range(50):
                    pred += model(input_x[0].reshape(1, 1, -1))
                pred /= 50
                feature = torch.concat(fea).mean(0)
                fea.clear()
            p.append(pred.detach().cpu().numpy())
            x.append(input_x.detach().cpu().numpy())
            y.append(input_y.detach().cpu().numpy())
            f.append(feature.detach().cpu().numpy())
            functional.reset_net(model)

    torch.save(np.concatenate(p), path + f'{fold}p.pth')
    torch.save(np.concatenate(x), path + f'{fold}x.pth')
    torch.save(np.concatenate(y), path + f'{fold}y.pth')
    torch.save(np.concatenate(f), path + f'{fold}f.pth')

    return diff_x, diff_y


if __name__ == '__main__':
    # 创建解析器
    parser = argparse.ArgumentParser(description='Your program description')

    # 添加参数
    parser.add_argument('--test_flag', type=ast.literal_eval, default=True, help='Test flag')
    parser.add_argument('--ord', type=ast.literal_eval, default=True, help='Ordinal Classification')
    parser.add_argument('--display', type=ast.literal_eval, default=False, help='Display the confusion matrix')
    parser.add_argument('--comment', type=str, default="Test", help='Comment of this run')
    parser.add_argument('--n_epoch', type=int, default=40, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--model_path', type=str, default='models/', help='Model path')
    parser.add_argument('--data_path', type=str, default='./', help='Data path')

    # 解析参数
    args = parser.parse_args()

    # 使用参数
    test_flag = args.test_flag
    ord = args.ord
    display = args.display
    comment = args.comment
    n_epoch = args.n_epoch
    batch_size = args.batch_size
    model_path = args.model_path
    data_path = args.data_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, 5))
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if test_flag:
        for i in range(5):
            c = comment + 'Fold' + i.__str__()
            time = datetime.datetime.now().strftime("%F-%T").replace(":", "-")
            save_path = f'./snnlogs/{time}-{c}/'
            stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True)
            data = pd.read_pickle(data_path + 'fragment.pkl')
            encoder = LabelEncoder()
            encoder.fit(data['label'].to_numpy())
            # 遍历每个折
            for fold, (train_idx, test_idx) in enumerate(stratified_kfold.split(data.iloc[:, :-1], data.iloc[:, -1])):
                X_test = data.iloc[test_idx, :-1]

                net = torch.load(f'{model_path}OrdChaoticDistillCV52024-student-{fold}-fold_epoch_39_params_file.pth')

                test_res(net, data.iloc[test_idx], fold, path=f'test_res/{comment}/')

    else:

        c = comment
        time = datetime.datetime.now().strftime("%F-%T").replace(":", "-")
        save_path = f'./snnlogs/{time}-{c}/'
        stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True)
        data = pd.read_pickle(data_path + 'fragment.pkl')
        # 遍历每个折
        for fold, (train_idx, test_idx) in enumerate(stratified_kfold.split(data.iloc[:, :-1], data.iloc[:, -1])):
            train = data.iloc[train_idx, :]
            test = data.iloc[test_idx, :]

            teacher = loader.load_ResNet1D(ord=ord)
            net = COSNN(1, [48], 1, 31, 4 - int(ord), 100)
            writer = SummaryWriter(save_path)
            gt, p, aucs, out = cross_validation(net, teacher, fold, c, writer, train, test, ord=ord)

            cr = classification_report(gt, p, digits=5)
            if display:
                con_mat = confusion_matrix(gt, p)
                con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=0)[np.newaxis, :]  # 归一化
                con_mat_norm = np.around(con_mat_norm, decimals=2)
                annot = pd.DataFrame(con_mat_norm).applymap(lambda x: f"{x}")
                annot += pd.DataFrame(con_mat).applymap(lambda x: f"\n({x})")
                sns.heatmap(con_mat_norm, annot=annot, fmt="s", cmap='Blues').set(xlabel="Predict", ylabel="Truth")
                plt.show()

            print(cr)

            torch.save(gt, save_path + fold.__str__() + 'gt.pt')
            torch.save(out, save_path + fold.__str__() + 'out.pt')
            torch.save(p, save_path + fold.__str__() + 'pred.pt')
            with open(save_path + "class_report.txt", "a", encoding="utf-8") as f:
                print(f.write(cr + "\n"))
                for i in range(4):
                    print(f.write(aucs[i].__str__() + "\n"))
