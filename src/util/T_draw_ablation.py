import os
import sys
import copy
import math
import numpy as np
import matplotlib.pyplot as plt 
import re

def draw_bar(array,file_name,bar_name):
    size = array.shape[1]
    n = array.shape[0]
    total_width = 0.75
    width = total_width / n
    x = np.arange(size)

    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})
    plt.figure(figsize=(12, 10))
    
    color_set = ['mistyrose','lightpink','limegreen','mediumseagreen','lightblue','skyblue','steelblue']
    patterns = ['//','\\\\','||','--','oo','xx']
    patterns = ['/','\\','|','-','o','x']
    patterns = ['','\\','','-','o','x']
    #color_set = ['lightsalmon','salmon','limegreen','mediumseagreen','aquamarine','turquoise']
    color_set = ['lightsalmon','salmon','lemonchiffon','khaki','lightblue','skyblue']

    for i in range(n):
        plt.bar(x+i*width,array[i],width=width*7/10,label=bar_name[i],color=color_set[i],hatch=patterns[i],edgecolor='black')
    
    if file_name.find("mape")!=-1:
        yname = "MAPE(%)"
    else:
        yname = r"Correlation Coefficient $R$"
    plt.ylabel(yname, fontsize=22)
    #plt.ylim(ymin=0)
    if yname == r"Correlation Coefficient $R$":
        plt.ylim(ymin=0.88,ymax=1)

    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    plt.xlabel('# Training Config', fontsize=22)
    if len(bar_name)>6:
        legend = plt.legend(fontsize=14, ncol=4,columnspacing=1, handlelength=2,
                    labelspacing=0.5, borderaxespad=0.5, loc='upper left',bbox_to_anchor=(0.05,1.17))
    else:
        legend = plt.legend(fontsize=14, ncol=3,columnspacing=1, handlelength=2,
                    labelspacing=0.5, borderaxespad=0.5, loc='upper left',bbox_to_anchor=(0.05,1.17))
                    
    legend.get_frame().set_linewidth(5)
    ax = plt.gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.5)
    plt.savefig(file_name+".jpg",dpi=200)
    plt.clf()
    plt.close()

mcpatcalib_mape = np.load("0_mapct_calib_mape.npy")
mcpatcalib_r = np.load("0_mapct_calib_r.npy")
mcpatransfer_mape = np.load("0_mapctransfer_mape.npy")
mcpatransfer_r = np.load("0_mapctransfer_r.npy")

mcpatcalib_transfer_mape = np.load("1_mapct_calib_transfer_mape.npy")
mcpatcalib_transfer_r = np.load("1_mapct_calib_transfer_r.npy")
comp_mape = np.load("2_comp_mape.npy")
comp_r = np.load("2_comp_r.npy")
comp_transfer_mape = np.load("3_comp_transfer_mape.npy")
comp_transfer_r = np.load("3_comp_transfer_r.npy")
resfunc_transfer_mape = np.load("4_resfunc_transfer_mape.npy")
resfunc_transfer_r = np.load("4_resfunc_transfer_r.npy")

mape = [mcpatcalib_mape,comp_mape,mcpatcalib_transfer_mape,comp_transfer_mape,resfunc_transfer_mape,mcpatransfer_mape]
r = [mcpatcalib_r,comp_r,mcpatcalib_transfer_r,comp_transfer_r,resfunc_transfer_r,mcpatransfer_r]

mape_XS2BOOM = 100*np.vstack(mape)[:,[5,3,1]]
mape_BOOM2XS = 100*np.vstack(mape)[:,[4,2,0]]

print((np.average(mape_XS2BOOM[0]-mape_XS2BOOM[5])+np.average(mape_BOOM2XS[0]-mape_BOOM2XS[5]))/2)

r_XS2BOOM = np.vstack(r)[:,[5,3,1]]
r_BOOM2XS = np.vstack(r)[:,[4,2,0]]

print((np.average(r_XS2BOOM[5]-r_XS2BOOM[0])+np.average(r_BOOM2XS[5]-r_BOOM2XS[0]))/2)

exit()
draw_bar(mape_XS2BOOM,'mape_XS2BOOM',["McPAT-Calib","McPAT-Calib+Comp","McPAT-Calib+Transfer","McPAT-Calib+Comp+Transfer","FirePower w/o Retraining","FirePower"])
draw_bar(mape_BOOM2XS,'mape_BOOM2XS',["McPAT-Calib","McPAT-Calib+Comp","McPAT-Calib+Transfer","McPAT-Calib+Comp+Transfer","FirePower w/o Retraining","FirePower"])
draw_bar(r_XS2BOOM,'r_XS2BOOM',["McPAT-Calib","McPAT-Calib+Comp","McPAT-Calib+Transfer","McPAT-Calib+Comp+Transfer","FirePower w/o Retraining","FirePower"])
draw_bar(r_BOOM2XS,'r_BOOM2XS',["McPAT-Calib","McPAT-Calib+Comp","McPAT-Calib+Transfer","McPAT-Calib+Comp+Transfer","FirePower w/o Retraining","FirePower"])