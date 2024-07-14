import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, ConfusionMatrixDisplay, \
    roc_curve, auc, accuracy_score
from statannotations.Annotator import Annotator

import CAM
from CAM import cam_test
from spikingjelly.activation_based import functional
from COSNN import COSNN
from scipy.stats import f_oneway
from utils import cal_pred_resT, cal_pred_res
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

color_map = ['#80C680', '#FFEC66', '#FFB26E', '#E67D7E']
colors = ['tab:green', '#ffdf00', 'tab:orange', 'tab:red']
boxcolors = ['g', 'g', '#ffdf00', '#ffdf00', '#ffdf00', '#ffdf00', '#ffdf00', '#ffdf00', 'tab:orange', 'tab:orange',
             'tab:orange', 'tab:orange', 'r', 'r', 'r', 'r']
ecg_risk_levels = {
    0: ["N", "Ne"],
    1: ["AFIB", "SVTA", "SBR", "BI", "NOD", "BBB"],
    2: ["VTLR", "VB", "HGEA", "VER"],
    3: ["VTHR", "VTTdP", "VFL", "VF"]
}

number_to_label = {
    0: "Normal",
    1: "Ne",
    2: "N",
    3: "AFIB",
    4: "SVTA",
    5: "SBR",
    6: "BI",
    7: "NOD",
    8: "BBB",
    9: "SEB",
    10: "VTLR",
    11: "B",
    12: "HGEA",
    13: "VER",
    14: "VEB",
    15: "VTHR",
    16: "VTTdP",
    17: "VFL",
    18: "VF",
    19: "3"
}


def label_trans(label):
    label_ret = []
    ecg_risk_levels = {
        0: ["N", "Ne", 'Normal', '0', 0],
        1: ["AFIB", "SVTA", "SBR", "BI", "NOD", "BBB", 'SEB', '1', 1],
        2: ["VTLR", "VB", "HGEA", "VER", '2', 'VEB', 2],
        3: ["VTHR", "VTTdP", "VFL", "VF", '3', 3]
    }
    ecg_abbr_to_risk = {ecg_abbr: risk for risk, ecg_list in ecg_risk_levels.items() for ecg_abbr in ecg_list}
    print(ecg_abbr_to_risk)
    for l in label:
        label_ret.append(ecg_abbr_to_risk[l])
    distribute = pd.Series(label_ret)
    print(distribute.value_counts())
    return label_ret


def load_data():
    pred_arr = torch.load('pred_arr.pth')
    gt_arr = torch.load('gt_arr.pth')
    return pred_arr, gt_arr


def preprocess_data(pred_arr, gt_arr):
    prob_arr = cal_pred_res(pred_arr)
    label_arr = [number_to_label.get(code) for code in gt_arr]
    grade_arr = label_trans(label_arr)
    return prob_arr, label_arr, grade_arr


def load_analyzed_data():
    p5 = torch.load('test_res/NCOKD/4p.pth')
    p4 = torch.load('test_res/NCOKD/3p.pth')
    p3 = torch.load('test_res/NCOKD/2p.pth')
    p2 = torch.load('test_res/NCOKD/1p.pth')
    p1 = torch.load('test_res/NCOKD/0p.pth')
    y5 = torch.load('test_res/NCOKD/4y.pth')
    y4 = torch.load('test_res/NCOKD/3y.pth')
    y3 = torch.load('test_res/NCOKD/2y.pth')
    y2 = torch.load('test_res/NCOKD/1y.pth')
    y1 = torch.load('test_res/NCOKD/0y.pth')
    x5 = torch.load('test_res/NCOKD/4x.pth')
    x4 = torch.load('test_res/NCOKD/3x.pth')
    x3 = torch.load('test_res/NCOKD/2x.pth')
    x2 = torch.load('test_res/NCOKD/1x.pth')
    x1 = torch.load('test_res/NCOKD/0x.pth')
    f5 = torch.load('test_res/NCOKD/4f.pth')
    f4 = torch.load('test_res/NCOKD/3f.pth')
    f3 = torch.load('test_res/NCOKD/2f.pth')
    f2 = torch.load('test_res/NCOKD/1f.pth')
    f1 = torch.load('test_res/NCOKD/0f.pth')
    label_arr = torch.load('analyse/new_label_arr.pth')
    prob_arr = torch.load('analyse/new_prob_arr.pth')
    grade_arr = torch.load('analyse/new_grade_arr.pth')
    boxcolors = torch.load('analyse/color_groups.pth')
    return label_arr, prob_arr, grade_arr, boxcolors


def plot_boxplot(prob_arr, grade_arr):
    label_x = []
    label_subgroups = []
    for i in range(4):
        tem = prob_arr[grade_arr == i][i]
        print(tem.shape)
        if not tem.shape == (0,):
            label_x.append(f"G{i}")
            label_subgroups.append(tem)
    box_plot = plt.boxplot(label_subgroups, flierprops=dict(marker='.', markersize=1, linestyle='none'), vert=False,
                           patch_artist=True)
    for box, color in zip(box_plot['boxes'], boxcolors):
        box.set_facecolor(color)
        box.set_alpha(0.6)
    plt.xlabel('Probability')
    plt.yticks(range(0, len(label_x) + 1), label_x)
    legend_labels = [("G0", color_map[0]), ("G1", color_map[1]), ("G2", color_map[2]), ("G3", color_map[3])]
    plt.legend([plt.Rectangle((0, 0), 1, 1, fc=color, edgecolor='black') for label, color in legend_labels],
               [label for label, color in legend_labels],
               bbox_to_anchor=(0.5, -0.2),
               loc='lower center', ncol=4, frameon=False)
    plt.show()


def perform_anova_test(prob_arr, label_arr):
    classes = [[] for _ in range(4)]
    for i in range(4):
        for l in ecg_risk_levels[i]:
            tem = prob_arr[label_arr == l][i]
            classes[i].append(list(tem.values))
    fvalue, pvalue = f_oneway(*classes)
    print("F-value:", fvalue)
    print("P-value:", pvalue)


def plot_roc_curve(y, p, grade_arr, prob_arr):
    plt.figure(figsize=(6, 4), dpi=180)
    color_list = colors
    ax = plt.gca()

    for i in range(4):
        fpr, tpr, thresholds = roc_curve((grade_arr == i), y[i][0], pos_label=1)
        roc_auc = auc(fpr, tpr)

        # 寻找最大值点
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        optimal_fpr, optimal_tpr = fpr[optimal_idx], tpr[optimal_idx]

        # 绘制 ROC 曲线及最大值点
        plt.plot(fpr, tpr, color=color_list[i], lw=2, label='${G_%d}$(AUC=%.4f)' % (i, roc_auc))
        plt.scatter(optimal_fpr, optimal_tpr, color=color_list[i])

    plt.plot([0, 1], [0, 1], color='tab:purple', lw=2, linestyle='--')

    # 设置坐标轴
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.legend(loc='lower right')
    plt.tight_layout()

    # 创建局部放大图
    axins = zoomed_inset_axes(ax, 2, loc='center')
    for i in range(4):
        fpr, tpr, thresholds = roc_curve((grade_arr == i), y[i][0], pos_label=1)
        optimal_idx = np.argmax(tpr - fpr)
        axins.plot(fpr, tpr, color=color_list[i])
        axins.scatter(fpr[optimal_idx], tpr[optimal_idx], color=color_list[i])

    # 设置局部放大图的坐标轴范围
    axins.set_xlim(0.01, 0.2)
    axins.set_ylim(0.75, 0.99)
    axins.set_xticks([])
    axins.set_yticks([])
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")
    plt.show()


def main():
    pred_arr, gt_arr = load_data()
    prob_arr, label_arr, grade_arr = preprocess_data(pred_arr, gt_arr)
    plot_boxplot(prob_arr, grade_arr)
    perform_anova_test(prob_arr, label_arr)


if __name__ == "__main__":
    main()
