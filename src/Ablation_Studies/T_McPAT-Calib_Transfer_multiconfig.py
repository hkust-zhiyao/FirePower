import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import copy
from sklearn.metrics import r2_score
from sklearn.metrics import pairwise 
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt


# Total OtherLogic 
# RNU Itlb Dtlb Regfile ROB IFU LSU FU_Pool 
# ICacheTagArray ICacheDataArray ICacheOther
# DCacheTagArray DCacheDataArray DCacheMSHR DCacheOther
# BPTAGE BPBTB BPOther
# ISUInt ISUMem ISUFp
# 2 + 8 + 3 + 4 + 3 + 3 = 23

r_list = []
mape_list = []

def load_data(uarch):
    feature = np.load('../../data/{}_panda_feature.npy'.format(uarch))
    label = np.load('../../data/{}_panda_label_fine_grained.npy'.format(uarch))
    print(label.shape)
    return feature, label

def train_model(mod, train_mod_feature, train_mod_label):
    mod.fit(train_mod_feature,train_mod_label)
    return mod

def build_model_for_one_component(build_feat, build_label):
    #res_function = res_model.predict(res_feat)
    #build_transformed_label = build_label / res_function
    
    model = xgb.XGBRegressor()
    trained_model = train_model(model,build_feat,build_label)
    return trained_model

def build_model(feature,label):
            
    # get respective feature and label
    start_event = 14
    end_event = 101
    feature_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13] + [item for item in range(start_event,end_event)] + [item for item in range(164,202)]
    component_feature = feature[:,feature_index]
    component_label = label
            
    # build model
    model = build_model_for_one_component(component_feature,component_label)

    return model
    
def predict(model,test_feature):
    # get respective feature and label
    start_event = 14
    end_event = 101
    feature_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13] + [item for item in range(start_event,end_event)] + [item for item in range(164,202)]
    component_feature = test_feature[:,feature_index]
            
    # compute and accumulate power
    power_component = model.predict(component_feature)

    return power_component

def data_augment(source_model,target_feature,target_label,valid_index):
    pseudo_label = predict(source_model,target_feature)
    gt_feature = target_feature[valid_index]
    gt_label = target_label[valid_index]
    calibrated_pseudo_label = np.zeros(target_label.shape)
    calibrated_pseudo_label[valid_index] = gt_label
    for sample_idx in range(target_feature.shape[0]):
        if sample_idx in valid_index:
            continue

        start_event = 14
        end_event = 101
        feature_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13] + [item for item in range(start_event,end_event)] + [item for item in range(164,202)]
            
        diff = gt_feature[:,feature_index] - target_feature[sample_idx,feature_index]
        #print(diff.shape)
        diff = np.linalg.norm(diff,axis=1)
        similar_sample_idx = np.argmin(diff)
        calibrated_pseudo_label[sample_idx] = pseudo_label[sample_idx] * (gt_label[similar_sample_idx] / pseudo_label[valid_index[similar_sample_idx]])
        
    return target_feature, calibrated_pseudo_label

def draw_figure(gt,pd,name):
    plt.clf()
        
    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})
    plt.figure(figsize=(6, 5))
    min_value = 10000
    max_value = 0
    for i in range(gt.shape[0]):
        pred_value = pd[i]
        label_value = gt[i]
        if pred_value>label_value:
            min_sample=label_value
            max_sample=pred_value
        else:
            max_sample=label_value
            min_sample=pred_value
        if max_value<max_sample:
            max_value=max_sample
        if min_value>min_sample:
            min_value=min_sample
    plt.plot([0,max_value],[0,max_value],color='silver')
        
    #plt.plot([0.4,1.6],[0.4,1.6],color='silver')
    color_set = ['b','g','r','c','m','y','k','skyblue','olive','gray','coral','gold','peru','pink','cyan','']
    for i in range(gt.shape[0]//8):
        x = gt[i*8:(i+1)*8]
        y = pd[i*8:(i+1)*8]
        plt.scatter(x,y,marker='.',color=color_set[i],label="C{}".format(i),alpha=0.5,s=160)

    # legend = plt.legend(fontsize=19, ncol=1, columnspacing=0.5, 
    #            labelspacing=0.4, borderaxespad=0.2, loc='upper left')
    # legend.get_frame().set_linewidth(5)

    plt.xlabel('Ground Truth (W)', fontsize=22)
    plt.ylabel('Prediction (W)', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
            
    r_report = np.corrcoef(gt,pd)[1][0]
    mape_report = mean_absolute_percentage_error(gt,pd)
    print(name)
    print("R = {}".format(r_report))
    print("MAPE = {}%".format(mape_report * 100))  

    r_list.append(r_report)
    mape_list.append(mape_report)
        
    plt.text(0,max_value/7*6,"MAPE={:.2f}%\nR={:.2f}".format(mape_report*100,r_report),fontsize=20,bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='silver',lw=5 ,alpha=0.7))
    ax = plt.gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.5)
    plt.savefig("McPAT-Calib_Transfer_multiconfig/{}.jpg".format(name),dpi=200)
    plt.close()
    return

def known_n_config(known,target_uarch,source_uarch):
    
    source_feature, source_label = load_data(source_uarch)
    target_feature, target_label = load_data(target_uarch)
    
    num_of_workload = 8
    num_of_config = target_feature.shape[0] // num_of_workload

    testing_config = [item for item in range(0,num_of_config-1)]
    training_config = [num_of_config-1]
    for i in range(known-1):
        training_config.append(int(i*(num_of_config/known-1)))
        testing_config.remove(int(i*(num_of_config/known-1)))
    training_set = []
    for config_id in training_config:
        for i in range(8):
            training_set.append(config_id*8+i)
    testing_set = []
    for config_id in testing_config:
        for i in range(8):
            testing_set.append(config_id*8+i) 
       
    #print(len(training_set))
    #print(len(testing_set))
        
    source_model = build_model(source_feature, source_label[:,0])
    feature_augment, label_augment = data_augment(source_model,target_feature,target_label[:,0],training_set)
    target_model = build_model(feature_augment, label_augment)
    prediction = predict(target_model, target_feature[testing_set])

    label = target_label[:,0]        
    r_report = np.corrcoef(label[testing_set],prediction)[1][0]
    mape_report = mean_absolute_percentage_error(label[testing_set],prediction)
    #print(label[testing_set])
    draw_figure(label[testing_set],prediction,"McPAT-Calib_Transfer_{}_{}_from_{}".format(known,target_uarch,source_uarch))
    print("Known_{}_config".format(known))
    print("R = {}".format(r_report))
    print("MAPE = {}%".format(mape_report * 100))
    return r_report, mape_report


known_n_config(2,"XS","BOOM")
known_n_config(2,"BOOM","XS")
known_n_config(3,"XS","BOOM")
known_n_config(3,"BOOM","XS")
known_n_config(4,"XS","BOOM")
known_n_config(4,"BOOM","XS")

np.save("1_mapct_calib_transfer_r",np.array(r_list))
np.save("1_mapct_calib_transfer_mape",np.array(mape_list))