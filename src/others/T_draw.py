import numpy as np
import os
import matplotlib.pyplot as plt


def cut_for_mape(mape_array):
    threshold=35
    first=0
    for i in range(mape_array.shape[0]):
        if mape_array[i]>threshold:
            first=i
            break
    for i in range(first+1,mape_array.shape[0]):
        mape_array[i]=100
    return mape_array

def cut_for_r(r_array):
    threshold=0.5
    first=0
    for i in range(r_array.shape[0]):
        if r_array[i]<threshold:
            first=i
            break
    for i in range(first+1,r_array.shape[0]):
        r_array[i]=0
    return r_array

def draw_diff(cpu_name,figure_name):
    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})
    plt.figure(figsize=(10, 9))
    plt.xlabel('Number of Known Config (n)', fontsize=30)
    if figure_name=="mape":
        plt.ylabel('MAPE (%)', fontsize=30)
    else:
        plt.ylabel(r'Correlation $R$', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    
    if cpu_name == "boom":
        mcpat_plus = np.load("panda_mcpat_plus_curve_{}.npy".format(figure_name))
        baseline_fixed = np.load("panda_mcpat_calib_claimed_selection_curve_{}.npy".format(figure_name))
        component_model = np.load("T_Comp_BOOM_{}.npy".format(figure_name))
        our_model = np.load("T_BOOM_{}.npy".format(figure_name))
        t_model = np.load("T_BOOM_from_XS_{}.npy".format(figure_name))
        t_comp = np.load("T_Comp_BOOM_from_XS_{}.npy".format(figure_name))
        t_auto = np.load("T_Auto_BOOM_from_XS_{}.npy".format(figure_name))
    else:
        mcpat_plus = np.load("xs_mcpat_plus_curve_{}.npy".format(figure_name))
        baseline_fixed = np.load("xs_mcpat_calib_claimed_selection_curve_{}.npy".format(figure_name))
        component_model = np.load("T_Comp_XS_{}.npy".format(figure_name))
        our_model = np.load("T_XS_{}.npy".format(figure_name))
        t_model = np.load("T_XS_from_BOOM_{}.npy".format(figure_name))
        t_comp = np.load("T_Comp_XS_from_BOOM_{}.npy".format(figure_name))
        t_auto = np.load("T_Auto_XS_from_BOOM_{}.npy".format(figure_name))
        
    
    if figure_name=="mape":
        our_model=our_model*100
        t_model=t_model*100
        component_model=component_model*100
        mcpat_plus=mcpat_plus*100
        baseline_fixed=baseline_fixed*100
        t_comp=t_comp*100
        t_auto=t_auto*100
        
    if cpu_name == "boom":
        x = np.array([i for i in range(1,15)])
    else:
        x = np.array([i for i in range(1,10)])
    
    plt.plot(x,mcpat_plus,color='lightgreen',linewidth=5,label="McPAT-plus",linestyle='--', zorder=1,marker='*', markersize=22)
    plt.plot(x,baseline_fixed,color='pink',linewidth=5,label="McPAT-Calib",linestyle='-.', zorder=5,marker='^', markersize=16)
    plt.plot(x,component_model,color="skyblue",linewidth=5,label="Component-level", zorder=6,marker='o', markersize=16)
    plt.plot(x,our_model,color='dodgerblue',linewidth=5,label="AutoPANDA", zorder=3,marker='o', markersize=16)
    plt.plot(x,t_model,color='orange',linewidth=5,label="TransferAuto",linestyle=':', zorder=3,marker='s', markersize=16)
    plt.plot(x,t_comp,color='gold',linewidth=5,label="TransferComp",linestyle=':', zorder=4,marker='s', markersize=16)
    plt.plot(x,t_auto,color='green',linewidth=5,label="TransferAuto++",linestyle='--', zorder=0,marker='*', markersize=22)
    
    if cpu_name == "boom":
        plt.xticks(np.arange(1, 15, 2), np.arange(14, 0, -2))
    else:
        plt.xticks(np.arange(1, 10, 2), np.arange(9, 0, -2))
    
    if figure_name=="mape":
        plt.ylim((0,24))
    else:
        plt.ylim((0.75,1.03))
    
    legend = plt.legend(fontsize=13, ncol=4, columnspacing=1, handlelength=2,
            labelspacing=0.2, borderaxespad=0.1, loc='upper left',bbox_to_anchor=(0,1.1))

    legend.get_frame().set_linewidth(5)
    ax = plt.gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.5)
    plt.savefig("{}_diff_unknown_{}.jpg".format(cpu_name,figure_name),dpi=200)

draw_diff("boom","mape")
draw_diff("boom","r")
draw_diff("xs","mape")
draw_diff("xs","r")