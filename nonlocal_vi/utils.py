import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx

### ploting
def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

def plot_data(data):
    fig, ax = plt.subplots()

    # define the colors
    cmap = mpl.colors.ListedColormap(['k', 'w'])

    # create a normalize object the describes the limits of
    # each color
    bounds = [0., 0.5, 1.]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(data, interpolation='none', cmap=cmap, norm=norm)
    plt.gca().invert_yaxis()
    set_size(5, 5)



def compute_implied_fdr(threshold, s):
    """
    Args:
        s: predicted probability
        threshold: level greater than threshold are selected

    Returns:
        fdr corresponding to the threshold
    """
    indicator = np.asarray([s > threshold])
    return np.sum((1 - s) * indicator) / np.sum(indicator)


def search_threshold(s, fdr):
    """
    Args:
        s: predicted probability
        fdr: controlled false discovery rate level

    Returns:
        largest threshold such that the fdr is less than the controlled level: fdr
    """
    for threshold in np.linspace(0, 1, 101):
        if compute_implied_fdr(threshold, s) < fdr:
            break
    return threshold

# def compute_selected(s,print_rslt='False', method='VI'):
#     loc = np.where(s>0)
#     if method == 'VI':
#         threshold = search_threshold(s, 0.05)
#         loc = np.where(s>threshold)
#     xy=[]
#     strength=[]
#     high_strength = {}
#     pos_strength = {}
#     for i in range(len(loc[0])):
#         pos = [loc[0][i], loc[1][i]]
#         nutrients_selected = str(nutrient_name[ffq_labels['id'] == str(nutrients[loc[0][i]]) ] )
#         pos_strength[nutrients_selected+ '+' + str(taxa[loc[1][i]])] = s[loc[0][i], loc[1][i]] * vmu_beta[loc[0][i], loc[1][i]]
#         xy.append(pos)
#         strength.append(s[loc[0][i], loc[1][i]] )
#         if s[loc[0][i], loc[1][i]] > 0.1:
#             high_strength[str(nutrients[loc[0][i]]) + '+' + str(taxa[loc[1][i]])] = s[loc[0][i], loc[1][i]]
#     if print_rslt == "True":
#         for i in pos_strength.items():
#             print(i)
#     return loc,pos_strength




class RealDataExp(object):
    def __init__(self, valpha, vmu_beta, data):
        self.beta = 2 / 3
        self.zeta = 1.1
        self.gamma = -0.1
        self.vmu_beta = vmu_beta
        self.pred = hard_sigmoid(sigmoid(valpha) * (self.zeta - self.gamma) + self.gamma)
        threshold = search_threshold(self.pred, 0.10)
        #         threshold = 0.5
        print('threshold', threshold)

        self.s = np.zeros_like(self.pred)
        self.s[self.pred < threshold] = 0
        self.s[self.pred > threshold] = 1

        self.nutrients = list(data['X'])  # short abbrevation of nutrients
        Y = pd.read_table('/extra/yadongl10/SVA/combo_count_tab_simplified.txt', sep=' ')
        self.taxa = list(Y)
        self.taxa = [self.taxa[i].split('.')[-1] for i in range(len(self.taxa))]
        self.ffq_labels = pd.read_table('ffq_labels2.txt', sep=" ",
                                        skiprows=1)  # map from nutrients abbrevation to full name
        self.nutrient_name = np.asarray(self.ffq_labels['Participant Identifier'])  # nutrients full name

    def compute_s_pred(self):
        return self.pred, self.s

    def plot_selected(self):
        print('selected association pattern (above plot)--DMVI')
        plot_data(self.s)
        print('selected association pattern (below plot)--ChenLigrouplasso')
        mean_beta_cl_real_data = np.abs(pd.read_csv('est_beta_cl_real_data_run2.txt', sep='\t').values[:, 2:])
        mean_beta_cl_real_data[mean_beta_cl_real_data <= 0.00] = 0
        mean_beta_cl_real_data[mean_beta_cl_real_data > 0.00] = 1
        mean_beta_cl_real_data = mean_beta_cl_real_data.T  # np.flip(mean_beta_cl_real_data.T, axis=1)
        plot_data(mean_beta_cl_real_data)

    def compute_selected(self, s, print_rslt='False', chenli=False):
        loc = np.where(s > 0)
        xy = []
        strength = []
        high_strength = {}
        pos_strength = {}
        for i in range(len(loc[0])):
            pos = [loc[0][i], loc[1][i]]
            nutrients_selected = str(self.nutrient_name[self.ffq_labels['id'] == str(self.nutrients[loc[0][i]])])
            pos_strength[nutrients_selected + '+' + str(self.taxa[loc[1][i]])] = self.pred[loc[0][i], loc[1][i]] * \
                                                                                 self.vmu_beta[
                                                                                     loc[0][i], loc[1][i]]
            if chenli == True:
                pos_strength[nutrients_selected + '+' + str(self.taxa[loc[1][i]])] = s[loc[0][i], loc[1][i]]
            xy.append(pos)
        if print_rslt == "True":
            for i in pos_strength.items():
                print(i)
        return loc, pos_strength

    def get_nutri_taxa_list(self, s):
        nutrient_list = []
        taxa_list = []
        loc = np.where(np.abs(s) > 0)
        for i in range(len(loc[0])):
            nutrient_list.append(self.nutrient_name[self.ffq_labels['id'] == str(self.nutrients[loc[0][i]])])
            taxa_list.append(self.taxa[loc[1][i]])

        nutrient_list = np.unique(nutrient_list).tolist()
        taxa_list = np.unique(taxa_list).tolist()
        return nutrient_list, taxa_list

    def draw_bipartite_graph(self, s, nutrient_list, taxa_list, chenli=False):
        G = nx.Graph()
        pos = nx.circular_layout(G)  # positions for all nodes
        loc = np.where(np.abs(s) > 0)
        plt.figure(figsize=(65, 50))
        # draw nodes
        G.add_nodes_from(nutrient_list)
        G.add_nodes_from(taxa_list)
        pos = nx.circular_layout(G)
        nx.draw_networkx_nodes(G, pos,
                               nodelist=nutrient_list,
                               node_color='green',
                               node_size=9500,
                               alpha=0.8)
        nx.draw_networkx_nodes(G, pos,
                               nodelist=taxa_list,
                               node_color='y',
                               node_shape='h',
                               node_size=9500,
                               alpha=0.8)

        for i in range(len(loc[0])):
            nutrients_selected = self.nutrient_name[
                self.ffq_labels['id'] == str(self.nutrients[loc[0][i]])]  # look up in the ffq_labels
            taxa_selected = [self.taxa[loc[1][i]]]
            # drasw edges
            strength = self.pred[loc[0][i], loc[1][i]] * self.vmu_beta[loc[0][i], loc[1][i]] * 3
            if chenli == True:
                strength = s[loc[0][i], loc[1][i]] * 500

            if strength > 0:
                edge_col = 'r'
                edge_style = 'solid'
            else:
                edge_col = 'blue'
                edge_style = 'dashed'
            nx.draw_networkx_edges(G, pos,
                                   edgelist=[(nutrients_selected[0], taxa_selected[0])],
                                   width=np.abs(strength), alpha=0.5, edge_color=edge_col, style=edge_style)

        pos_higher = {}
        y_off = .07  # offset on the y axis
        for k, v in pos.items():
            pos_higher[k] = (v[0], v[1] + y_off)
            if k == 'Pyridoxine B6 w/o vit. pills' or k == 'Pantothenic Acid w/o suppl.':
                print(v)
                pos_higher[k] = (v[0] + 0.15, v[1] + y_off)
        nx.draw_networkx_labels(G, pos_higher, font_size=60, font_family='sans-serif')
        plt.axis('off')
        plt.savefig('bipartite.pdf')
        print('plot saved')
        plt.show()

        return G


import torch
import torch.nn as nn
from torch.nn import Module

delta = 1e-6
softplus_ = nn.Softplus()
softplus = lambda x: softplus_(x) + delta
sigmoid_ = nn.Sigmoid()
sigmoid = lambda x: sigmoid_(x) * (1 - delta) + 0.5 * delta
sigmoid2 = lambda x: sigmoid(x) * 2.0
logsigmoid = lambda x: -softplus(-x)
logit = lambda x: torch.log
log = lambda x: torch.log(x * 1e2) - np.log(1e2)
logit = lambda x: log(x) - log(1 - x)


def softmax(x, dim=-1):
    e_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    out = e_x / e_x.sum(dim=dim, keepdim=True)
    return out


sum1 = lambda x: x.sum(1)
sum_from_one = lambda x: sum_from_one(sum1(x)) if len(x.size()) > 2 else sum1(x)


class Sigmoid(Module):
    def forward(self, x):
        return sigmoid(x)


def oper(array, oper, axis=-1, keepdims=False):
    a_oper = oper(array)
    if keepdims:
        shape = []
        for j,s in enumerate(array.size()):
            shape.append(s)
        shape[axis] = -1
        a_oper = a_oper.view(*shape)
    return a_oper


def log_sum_exp(A, axis=-1, sum_op=torch.sum):
    maximum = lambda x: x.max(axis)[0]
    A_max = oper(A,maximum,axis,True)
    summation = lambda x: sum_op(torch.exp(x-A_max), axis)
    B = torch.log(oper(A,summation,axis,True)) + A_max
    return B


import seaborn as sns


def plot(x, savefig=True):
    mean = np.asarray([0.6, 1.2, 1.8, 2.4, 3])
    f, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    for i in range(2):
        for j in range(3):
            if i == 1 and j == 2:
                continue
            theta = x[:, -(i * 3 + j + 1)]
            sns.distplot(theta, bins=50, ax=axes[i, j])
            axes[i, j].set_title('theta {}'.format((i * 3 + j + 1)), fontsize=16)
            axes[i, j].axvline(x=mean[-(i * 3 + j + 1)], color='red', linestyle='--')
    for ax in axes.flat:
        ax.set_xlabel('values', fontsize=15)
        ax.set_ylabel('density', fontsize=15)
    if savefig:
        plt.savefig('est_theta_density.png')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx

### ploting
def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

def plot_data(data):
    fig, ax = plt.subplots()

    # define the colors
    cmap = mpl.colors.ListedColormap(['k', 'w'])

    # create a normalize object the describes the limits of
    # each color
    bounds = [0., 0.5, 1.]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(data, interpolation='none', cmap=cmap, norm=norm)
    plt.gca().invert_yaxis()
    set_size(5, 5)



def compute_implied_fdr(threshold, s):
    """
    Args:
        s: predicted probability
        threshold: level greater than threshold are selected

    Returns:
        fdr corresponding to the threshold
    """
    indicator = np.asarray([s > threshold])
    return np.sum((1 - s) * indicator) / np.sum(indicator)


def search_threshold(s, fdr):
    """
    Args:
        s: predicted probability
        fdr: controlled false discovery rate level

    Returns:
        largest threshold such that the fdr is less than the controlled level: fdr
    """
    for threshold in np.linspace(0, 1, 101):
        if compute_implied_fdr(threshold, s) < fdr:
            break
    return threshold

# def compute_selected(s,print_rslt='False', method='VI'):
#     loc = np.where(s>0)
#     if method == 'VI':
#         threshold = search_threshold(s, 0.05)
#         loc = np.where(s>threshold)
#     xy=[]
#     strength=[]
#     high_strength = {}
#     pos_strength = {}
#     for i in range(len(loc[0])):
#         pos = [loc[0][i], loc[1][i]]
#         nutrients_selected = str(nutrient_name[ffq_labels['id'] == str(nutrients[loc[0][i]]) ] )
#         pos_strength[nutrients_selected+ '+' + str(taxa[loc[1][i]])] = s[loc[0][i], loc[1][i]] * vmu_beta[loc[0][i], loc[1][i]]
#         xy.append(pos)
#         strength.append(s[loc[0][i], loc[1][i]] )
#         if s[loc[0][i], loc[1][i]] > 0.1:
#             high_strength[str(nutrients[loc[0][i]]) + '+' + str(taxa[loc[1][i]])] = s[loc[0][i], loc[1][i]]
#     if print_rslt == "True":
#         for i in pos_strength.items():
#             print(i)
#     return loc,pos_strength




class RealDataExp(object):
    def __init__(self, valpha, vmu_beta, data):
        self.beta = 2 / 3
        self.zeta = 1.1
        self.gamma = -0.1
        self.vmu_beta = vmu_beta
        self.pred = hard_sigmoid(sigmoid(valpha) * (self.zeta - self.gamma) + self.gamma)
        threshold = search_threshold(self.pred, 0.10)
        #         threshold = 0.5
        print('threshold', threshold)

        self.s = np.zeros_like(self.pred)
        self.s[self.pred < threshold] = 0
        self.s[self.pred > threshold] = 1

        self.nutrients = list(data['X'])  # short abbrevation of nutrients
        Y = pd.read_table('/extra/yadongl10/SVA/combo_count_tab_simplified.txt', sep=' ')
        self.taxa = list(Y)
        self.taxa = [self.taxa[i].split('.')[-1] for i in range(len(self.taxa))]
        self.ffq_labels = pd.read_table('ffq_labels2.txt', sep=" ",
                                        skiprows=1)  # map from nutrients abbrevation to full name
        self.nutrient_name = np.asarray(self.ffq_labels['Participant Identifier'])  # nutrients full name

    def compute_s_pred(self):
        return self.pred, self.s

    def plot_selected(self):
        print('selected association pattern (above plot)--DMVI')
        plot_data(self.s)
        print('selected association pattern (below plot)--ChenLigrouplasso')
        mean_beta_cl_real_data = np.abs(pd.read_csv('est_beta_cl_real_data_run2.txt', sep='\t').values[:, 2:])
        mean_beta_cl_real_data[mean_beta_cl_real_data <= 0.00] = 0
        mean_beta_cl_real_data[mean_beta_cl_real_data > 0.00] = 1
        mean_beta_cl_real_data = mean_beta_cl_real_data.T  # np.flip(mean_beta_cl_real_data.T, axis=1)
        plot_data(mean_beta_cl_real_data)

    def compute_selected(self, s, print_rslt='False', chenli=False):
        loc = np.where(s > 0)
        xy = []
        strength = []
        high_strength = {}
        pos_strength = {}
        for i in range(len(loc[0])):
            pos = [loc[0][i], loc[1][i]]
            nutrients_selected = str(self.nutrient_name[self.ffq_labels['id'] == str(self.nutrients[loc[0][i]])])
            pos_strength[nutrients_selected + '+' + str(self.taxa[loc[1][i]])] = self.pred[loc[0][i], loc[1][i]] * \
                                                                                 self.vmu_beta[
                                                                                     loc[0][i], loc[1][i]]
            if chenli == True:
                pos_strength[nutrients_selected + '+' + str(self.taxa[loc[1][i]])] = s[loc[0][i], loc[1][i]]
            xy.append(pos)
        if print_rslt == "True":
            for i in pos_strength.items():
                print(i)
        return loc, pos_strength

    def get_nutri_taxa_list(self, s):
        nutrient_list = []
        taxa_list = []
        loc = np.where(np.abs(s) > 0)
        for i in range(len(loc[0])):
            nutrient_list.append(self.nutrient_name[self.ffq_labels['id'] == str(self.nutrients[loc[0][i]])])
            taxa_list.append(self.taxa[loc[1][i]])

        nutrient_list = np.unique(nutrient_list).tolist()
        taxa_list = np.unique(taxa_list).tolist()
        return nutrient_list, taxa_list

    def draw_bipartite_graph(self, s, nutrient_list, taxa_list, chenli=False):
        G = nx.Graph()
        pos = nx.circular_layout(G)  # positions for all nodes
        loc = np.where(np.abs(s) > 0)
        plt.figure(figsize=(65, 50))
        # draw nodes
        G.add_nodes_from(nutrient_list)
        G.add_nodes_from(taxa_list)
        pos = nx.circular_layout(G)
        nx.draw_networkx_nodes(G, pos,
                               nodelist=nutrient_list,
                               node_color='green',
                               node_size=9500,
                               alpha=0.8)
        nx.draw_networkx_nodes(G, pos,
                               nodelist=taxa_list,
                               node_color='y',
                               node_shape='h',
                               node_size=9500,
                               alpha=0.8)

        for i in range(len(loc[0])):
            nutrients_selected = self.nutrient_name[
                self.ffq_labels['id'] == str(self.nutrients[loc[0][i]])]  # look up in the ffq_labels
            taxa_selected = [self.taxa[loc[1][i]]]
            # drasw edges
            strength = self.pred[loc[0][i], loc[1][i]] * self.vmu_beta[loc[0][i], loc[1][i]] * 3
            if chenli == True:
                strength = s[loc[0][i], loc[1][i]] * 500

            if strength > 0:
                edge_col = 'r'
                edge_style = 'solid'
            else:
                edge_col = 'blue'
                edge_style = 'dashed'
            nx.draw_networkx_edges(G, pos,
                                   edgelist=[(nutrients_selected[0], taxa_selected[0])],
                                   width=np.abs(strength), alpha=0.5, edge_color=edge_col, style=edge_style)

        pos_higher = {}
        y_off = .07  # offset on the y axis
        for k, v in pos.items():
            pos_higher[k] = (v[0], v[1] + y_off)
            if k == 'Pyridoxine B6 w/o vit. pills' or k == 'Pantothenic Acid w/o suppl.':
                print(v)
                pos_higher[k] = (v[0] + 0.15, v[1] + y_off)
        nx.draw_networkx_labels(G, pos_higher, font_size=60, font_family='sans-serif')
        plt.axis('off')
        plt.savefig('bipartite.pdf')
        print('plot saved')
        plt.show()

        return G


import torch
import torch.nn as nn
from torch.nn import Module

delta = 1e-6
softplus_ = nn.Softplus()
softplus = lambda x: softplus_(x) + delta
sigmoid_ = nn.Sigmoid()
sigmoid = lambda x: sigmoid_(x) * (1 - delta) + 0.5 * delta
sigmoid2 = lambda x: sigmoid(x) * 2.0
logsigmoid = lambda x: -softplus(-x)
logit = lambda x: torch.log
log = lambda x: torch.log(x * 1e2) - np.log(1e2)
logit = lambda x: log(x) - log(1 - x)


def softmax(x, dim=-1):
    e_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    out = e_x / e_x.sum(dim=dim, keepdim=True)
    return out


sum1 = lambda x: x.sum(1)
sum_from_one = lambda x: sum_from_one(sum1(x)) if len(x.size()) > 2 else sum1(x)


class Sigmoid(Module):
    def forward(self, x):
        return sigmoid(x)


def oper(array, oper, axis=-1, keepdims=False):
    a_oper = oper(array)
    if keepdims:
        shape = []
        for j,s in enumerate(array.size()):
            shape.append(s)
        shape[axis] = -1
        a_oper = a_oper.view(*shape)
    return a_oper


def log_sum_exp(A, axis=-1, sum_op=torch.sum):
    maximum = lambda x: x.max(axis)[0]
    A_max = oper(A,maximum,axis,True)
    summation = lambda x: sum_op(torch.exp(x-A_max), axis)
    B = torch.log(oper(A,summation,axis,True)) + A_max
    return B


import seaborn as sns


def plot(x, savefig=True):
    mean = np.asarray([0.6, 1.2, 1.8, 2.4, 3])
    f, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    for i in range(2):
        for j in range(3):
            if i == 1 and j == 2:
                continue
            theta = x[:, -(i * 3 + j + 1)]
            sns.distplot(theta, bins=50, ax=axes[i, j])
            axes[i, j].set_title('theta {}'.format((i * 3 + j + 1)), fontsize=16)
            axes[i, j].axvline(x=mean[-(i * 3 + j + 1)], color='red', linestyle='--')
    for ax in axes.flat:
        ax.set_xlabel('values', fontsize=15)
        ax.set_ylabel('density', fontsize=15)
    if savefig:
        plt.savefig('est_theta_density.png')


def plot_theta_posterior(x, savefig=True):
    mean = np.asarray([1, 2, 3, 4, 5])
    f, axes = plt.subplots(1, 5, figsize=(18, 10), sharex=True)
    for i in range(5):
        theta = x[:, -(5-i)]
        sns.distplot(theta, bins=50, ax=axes[i])
        axes[i].set_title('theta {}'.format(i+1), fontsize=16)
        axes[i].axvline(x=mean[-(i + 1)], color='red', linestyle='--')
    for ax in axes.flat:
        ax.set_xlabel('values', fontsize=15)
        ax.set_ylabel('density', fontsize=15)
    if savefig:
        plt.savefig('est_theta_density.png')


def search_threshold(s, fdr):
    """
    Args:
        s: predicted probability
        fdr: controlled false discovery rate level

    Returns:
        largest threshold such that the fdr is less than the controlled level: fdr
    """
    for threshold in np.linspace(0, 1, 101):
        if compute_implied_fdr(threshold, s) < fdr:
            break
    return threshold



def plot(x, savefig=True):
    mean = np.asarray([1, 2, 3, 4, 5])
    f, axes = plt.subplots(1, 5, figsize=(18, 10), sharex=True)
    for i in range(2):
        for j in range(3):
            if i == 1 and j == 2:
                continue
            theta = x[:, -(i * 3 + j + 1)]
            sns.distplot(theta, bins=50, ax=axes[i, j])
            axes[i, j].set_title('theta {}'.format((i * 3 + j + 1)), fontsize=16)
            axes[i, j].axvline(x=mean[-(i * 3 + j + 1)], color='red', linestyle='--')
    for ax in axes.flat:
        ax.set_xlabel('values', fontsize=15)
        ax.set_ylabel('density', fontsize=15)
    if savefig:
        plt.savefig('est_theta_density.png')


def search_threshold(s, fdr):
    """
    Args:
        s: predicted probability
        fdr: controlled false discovery rate level

    Returns:
        largest threshold such that the fdr is less than the controlled level: fdr
    """
    for threshold in np.linspace(0, 1, 101):
        if compute_implied_fdr(threshold, s) < fdr:
            break
    return threshold

