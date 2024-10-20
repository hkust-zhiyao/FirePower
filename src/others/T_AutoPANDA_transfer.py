import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import copy
from sklearn.metrics import r2_score
from sklearn.metrics import pairwise 
from sklearn.metrics import mean_absolute_percentage_error


# Total OtherLogic 
# RNU Itlb Dtlb Regfile ROB IFU LSU FU_Pool 
# ICacheTagArray ICacheDataArray ICacheOther
# DCacheTagArray DCacheDataArray DCacheMSHR DCacheOther
# BPTAGE BPBTB BPOther
# ISUInt ISUMem ISUFp
# 2 + 8 + 3 + 4 + 3 + 3 = 23

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
    feature = np.load('{}_panda_feature.npy'.format(uarch))
    label = np.load('{}_panda_label_fine_grained.npy'.format(uarch))
    print(label.shape)
    return feature, label

def train_model(mod, train_mod_feature, train_mod_label):
    mod.fit(train_mod_feature,train_mod_label)
    return mod

def build_model_for_one_component(res_model, res_feat, build_feat, build_label):
    res_function = res_model.predict(res_feat)
    build_transformed_label = build_label / res_function
    
    model = xgb.XGBRegressor()
    trained_model = train_model(model,build_feat,build_transformed_label)
    return trained_model

def build_model(feature,label,res_model):
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
        ml_model_this_component = build_model_for_one_component(res_model[iter],res_feature,component_feature,component_label)
        model_list.append(ml_model_this_component)
        iter = iter + 1
    return model_list

def test_for_one_component(res_model, model, res_feat, test_feat):
    res_function = res_model.predict(res_feat)
    pred_part = model.predict(test_feat)
    power_pred = pred_part * res_function
    return power_pred
    
def predict_per_comp(res_model,model_list,test_feature):
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
        power_component = test_for_one_component(res_model[iter],model_component,res_feature,component_feature)
        power_list.append(power_component)
            
        iter = iter + 1
            
    return power_list

def predict(model_list,res_model,test_feature):
    power_list = predict_per_comp(res_model,model_list,test_feature)
    power_value = np.zeros(test_feature.shape[0])
    for power in power_list:
        power_value = power_value + power      
    return power_value

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

def data_augment(source_model,target_feature,target_label,valid_index,res_model):
    pseudo_label_list = predict_per_comp(res_model,source_model,target_feature)
    gt_feature = target_feature[valid_index]
    gt_label = target_label[valid_index]
    calibrated_pseudo_label = np.zeros(target_label.shape)
    calibrated_pseudo_label[valid_index] = gt_label
    for sample_idx in range(target_feature.shape[0]):
        if sample_idx in valid_index:
            continue
        iter = 0
        for component in event_feature_of_components.keys():
            
            start_event = event_feature_of_components[component][0]
            end_event = event_feature_of_components[component][1]
            feature_index = params_feature_of_components[component] + [item for item in range(start_event,end_event)]
            
            diff = gt_feature[:,feature_index] - target_feature[sample_idx,feature_index]
            diff = np.linalg.norm(diff,axis=1)
            similar_sample_idx = np.argmin(diff)
            calibrated_pseudo_label[sample_idx,iter+1] = pseudo_label_list[iter][sample_idx] * (gt_label[similar_sample_idx,iter+1] / pseudo_label_list[iter][valid_index[similar_sample_idx]])
            iter = iter + 1
            
        calibrated_pseudo_label[sample_idx,0] = np.sum(calibrated_pseudo_label[sample_idx,1:])
    return target_feature, calibrated_pseudo_label

def unknown_n_config(unknown,target_uarch,source_uarch):
    
    source_feature, source_label = load_data(source_uarch)
    target_feature, target_label = load_data(target_uarch)
    
    num_of_workload = 8
    num_of_config = target_feature.shape[0] // num_of_workload
    
    fold = num_of_config
    test_size = num_of_workload * unknown
    pred_acc_vector = np.zeros((num_of_config*num_of_workload))
        
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
        
        res_model = get_resource_function(source_feature, source_label)
        source_model = build_model(source_feature, source_label, res_model)
        feature_augment, label_augment = data_augment(source_model,target_feature,target_label,training_set,res_model)
        #augmented_res_model = res_augment(res_model,target_feature,target_label,training_set)
        target_model = build_model(feature_augment, label_augment, res_model)
        prediction = predict(target_model, res_model, target_feature[testing_set])

        pred_acc_vector[testing_set] = pred_acc_vector[testing_set] + prediction
        
    pred_acc_vector = pred_acc_vector / unknown
    label = target_label[:,0]
    
    r_report = np.corrcoef(label,pred_acc_vector)[1][0]
    mape_report = mean_absolute_percentage_error(label,pred_acc_vector)
    print("Unknown_{}_config".format(unknown))
    print("R = {}".format(r_report))
    print("MAPE = {}%".format(mape_report * 100))
    return r_report, mape_report


curve_mape = []
curve_r = []

for i in range(9,10):
    r,mape = unknown_n_config(i,"XS","BOOM")
    #unknown_n_config(i,"BOOM","XS")
    curve_mape.append(mape)
    curve_r.append(r)

np.save("T_Auto_XS_from_BOOM_mape.npy",np.array(curve_mape))
np.save("T_Auto_XS_from_BOOM_r.npy",np.array(curve_r))
    
curve_mape = []
curve_r = []    

for i in range(14,15):
    r,mape = unknown_n_config(i,"BOOM","XS")
    #unknown_n_config(i,"BOOM","XS")
    curve_mape.append(mape)
    curve_r.append(r)
    
np.save("T_Auto_BOOM_from_XS_mape.npy",np.array(curve_mape))
np.save("T_Auto_BOOM_from_XS_r.npy",np.array(curve_r))