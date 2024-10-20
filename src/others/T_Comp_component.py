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

figure_name = ["Total", "OtherLogic",
"RNU", "Itlb", "Dtlb", "Regfile", "ROB", "IFU", "LSU", "FU_Pool", 
"ICacheTagArray", "ICacheDataArray", "ICacheOther",
"DCacheTagArray", "DCacheDataArray", "DCacheMSHR", "DCacheOther",
"BPTAGE", "BPBTB", "BPOther",
"ISUInt", "ISUMem", "ISUFp"]

# each pair represents the start and end points of the events related to each component 
event_feature_of_components={
            "OtherLogic":[14,101],
            "RNU":[121,128],
            "Itlb":[128,130],
            "Dtlb":[130,132],
            "Regfile":[132,137],
            "ROB":[137,139],
            "IFU":[139,156],
            "LSU":[156,159],
            "FU_Pool":[159,161],
            "ICacheTagArray":[111,117],
            "ICacheDataArray":[111,117],
            "ICacheOther":[111,117],
            "DCacheTagArray":[101,111],
            "DCacheDataArray":[101,111],
            "DCacheMSHR":[101,111],
            "DCacheOther":[101,111],
            "BPTAGE":[117,121],
            "BPBTB":[117,121],
            "BPOther":[117,121],
            "ISUInt":[161,164],
            "ISUMem":[161,164],
            "ISUFp":[161,164]
        }
        
# each list represents the configuration parameters related to each component
params_feature_of_components={
            "OtherLogic":[0,1,2,3,4,5,6,7,8,9,10,11,12,13],
            "RNU":[1],
            "Itlb":[1],
            "Dtlb":[11],
            "Regfile":[1,4,5],
            "ROB":[1,3],
            "IFU":[0,1,2,13],
            "LSU":[6,8],
            "FU_Pool":[8,9],
            "ICacheTagArray":[10,13],
            "ICacheDataArray":[10,13],
            "ICacheOther":[10,13],
            "DCacheTagArray":[8,10,11,12],
            "DCacheDataArray":[8,10,11,12],
            "DCacheMSHR":[8,10,11,12],
            "DCacheOther":[8,10,11,12],
            "BPTAGE":[0,1,7],
            "BPBTB":[0,1,7],
            "BPOther":[0,1,7],
            "ISUInt":[1,8,9],
            "ISUMem":[1,8,9],
            "ISUFp":[1,8,9]
        }

component_model_selection=[1,0,0,0,0,0,0,1,1,0,1,0,1]

def load_data(uarch):
    feature = np.load('npy/{}_panda_feature.npy'.format(uarch))
    label = np.load('npy/{}_panda_label_fine_grained.npy'.format(uarch))
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
    model_list = []
    iter = 0
    for component in event_feature_of_components.keys():
            
        # get respective feature and label
        start_event = event_feature_of_components[component][0]
        end_event = event_feature_of_components[component][1]
        feature_index = params_feature_of_components[component] + [item for item in range(start_event,end_event)]
        component_feature = feature[:,feature_index]
        res_feature = feature[:,params_feature_of_components[component]]
        label_index = iter + 1
        component_label = label[:,label_index]
            
        # build model
        ml_model_this_component = build_model_for_one_component(component_feature,component_label)
        model_list.append(ml_model_this_component)
        iter = iter + 1
    return model_list

def test_for_one_component(model, test_feat):
    #res_function = res_model.predict(res_feat)
    pred_part = model.predict(test_feat)
    #power_pred = pred_part * res_function
    return pred_part
    
def predict_per_comp(model_list,test_feature):
    iter = 0
    power_list = []
    for component in event_feature_of_components.keys():
        # get model
        model_component = model_list[iter]
            
        # get respective feature and label
        start_event = event_feature_of_components[component][0]
        end_event = event_feature_of_components[component][1]
        feature_index = params_feature_of_components[component] + [item for item in range(start_event,end_event)]
        
        #print(feature_index)
        #print(test_feature.shape)
        component_feature = test_feature[:,feature_index]
        res_feature = test_feature[:,params_feature_of_components[component]]
            
        # compute and accumulate power
        power_component = test_for_one_component(model_component,component_feature)
        power_list.append(power_component)
            
        iter = iter + 1
            
    return power_list

def predict(model_list,test_feature):
    power_list = predict_per_comp(model_list,test_feature)
    power_value = np.zeros(test_feature.shape[0])
    for power in power_list:
        power_value = power_value + power      
    return np.vstack([power_value] + power_list).T

def get_resource_function(source_feature, source_label):
    res_model = []
    iter = 0
    
    for component in event_feature_of_components.keys():
        feature_index = params_feature_of_components[component]
        feature = source_feature[:,feature_index]
        feature = feature.reshape((source_feature.shape[0]//8,8,feature.shape[1]))
        feature = np.average(feature,axis=1)
        
        label_index = iter + 1
        label = source_label[:,label_index]
        label = label.reshape((source_label.shape[0]//8,8))
        label = np.average(label,axis=1)
        
        model = xgb.XGBRegressor()
        res_model_this_component = train_model(model,feature,label)
        res_model.append(res_model_this_component)
        iter = iter + 1
        
    return res_model

def res_augment(res_model,target_feature,target_label,valid_index):
    return res_model

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
        
    plt.text(0,max_value/7*6,"MAPE={:.2f}%\nR={:.2f}".format(mape_report*100,r_report),fontsize=20,bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='silver',lw=5 ,alpha=0.7))
    ax = plt.gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.5)
    plt.savefig("T_Comp_result/{}.jpg".format(name),dpi=200)
    plt.close()
    return

def unknown_n_config(unknown,target_uarch,source_uarch):
    
    source_feature, source_label = load_data(source_uarch)
    target_feature, target_label = load_data(target_uarch)
    
    num_of_workload = 8
    num_of_config = target_feature.shape[0] // num_of_workload
    
    fold = num_of_config
    test_size = num_of_workload * unknown
    #pred_acc_vector = np.zeros((num_of_config*num_of_workload))
    pred_acc_vector = np.zeros((num_of_config*num_of_workload,23))
        
    for i in range(fold):
        start_point = num_of_workload*i
        end_point = start_point + test_size
        if end_point <= target_feature.shape[0]:
            testing_set = [item for item in range(start_point,end_point)]
            training_set = [item for item in range(0,start_point)] + [item for item in range(end_point,target_feature.shape[0])]
        else:
            end_point = end_point - target_feature.shape[0]
            testing_set = [item for item in range(start_point,target_feature.shape[0])] + [item for item in range(0,end_point)]
            training_set = [item for item in range(end_point,start_point)]
        
        #res_model = get_resource_function(source_feature, source_label)
        #augmented_res_model = res_augment(res_model,target_feature,target_label,training_set)
        target_model = build_model(target_feature[training_set], target_label[training_set])
        prediction = predict(target_model, target_feature[testing_set])

        pred_acc_vector[testing_set] = pred_acc_vector[testing_set] + prediction
        
    pred_acc_vector = pred_acc_vector / unknown
    for i in range(23):
        draw_figure(target_label[:,i],pred_acc_vector[:,i],figure_name[i]+"_{}_from_{}".format(target_uarch,source_uarch))
    label = target_label[:,0]
        
    # r_report = np.corrcoef(label,pred_acc_vector)[1][0]
    # mape_report = mean_absolute_percentage_error(label,pred_acc_vector)
    # print("Unknown_{}_config".format(unknown))
    # print("R = {}".format(r_report))
    # print("MAPE = {}%".format(mape_report * 100))
    return # r_report, mape_report


curve_mape = []
curve_r = []

for i in range(9,10):
    unknown_n_config(i,"XS","BOOM")
    #unknown_n_config(i,"BOOM","XS")
    # curve_mape.append(mape)
    # curve_r.append(r)

# np.save("T_Comp_XS_mape.npy",np.array(curve_mape))
# np.save("T_Comp_XS_r.npy",np.array(curve_r))
    
curve_mape = []
curve_r = []    

for i in range(14,15):
    unknown_n_config(i,"BOOM","XS")
    #unknown_n_config(i,"BOOM","XS")
    # curve_mape.append(mape)
    # curve_r.append(r)
    
# np.save("T_Comp_BOOM_mape.npy",np.array(curve_mape))
# np.save("T_Comp_BOOM_r.npy",np.array(curve_r))