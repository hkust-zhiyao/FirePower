import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import copy
from sklearn.metrics import r2_score
from sklearn.metrics import pairwise 
from sklearn.metrics import mean_absolute_percentage_error

# each pair represents the start and end points of the events related to each component 
event_feature_of_components={
            "OtherLogic":[14,101],
            "DCache":[101,111],
            "ICache":[111,117],
            "BP":[117,121],
            "RNU":[121,128],
            "Itlb":[128,130],
            "Dtlb":[130,132],
            "Regfile":[132,137],
            "ROB":[137,139],
            "IFU":[139,156],
            "LSU":[156,159],
            "FU_Pool":[159,161],
            "ISU":[161,164]
        }
        
# each list represents the configuration parameters related to each component
params_feature_of_components={
            "OtherLogic":[0,1,2,3,4,5,6,7,8,9,10,11,12,13],
            "DCache":[8,10,11,12],
            "ICache":[10,13],
            "BP":[0,7],
            "RNU":[1],
            "Itlb":[1],
            "Dtlb":[11],
            "Regfile":[1,4,5],
            "ROB":[1,3],
            "IFU":[0,1,2,13],
            "LSU":[6,8],
            "FU_Pool":[8,9],
            "ISU":[1,8,9]
        }
        
# each list represents which configuration parameters (in the list above) should be considered in the resource function, for itlb and fu_pool, no parameter is considered
encode_table={
            "BP":[0],
            "ICache":[0,1],
            "DCache":[0,1],
            "ISU":[0],
            "OtherLogic":[1],
            "IFU":[1],
            "ROB":[1],
            "Regfile":[1,2],
            "RNU":[0],
            "Dtlb":[0],
            "LSU":[0]            
        }
        
        # which model is selected for each component, 0 represents xgboost, 1 represents gradientboost
component_model_selection=[1,0,0,0,0,0,0,1,1,0,1,0,1]

logic_bias = 0
dtlb_bias = 0

def load_data(uarch):
    feature = np.load('{}_panda_feature.npy'.format(uarch))
    label = np.load('{}_panda_label.npy'.format(uarch))
    return feature, label

def compute_reserve_station_entries(decodewidth_init):
    decodewidth = int(decodewidth_init+0.01)
    isu_params = [
            # IQT_MEM.numEntries IQT_MEM.dispatchWidth
            # IQT_INT.numEntries IQT_INT.dispatchWidth
            # IQT_FP.numEntries IQT_FP.dispatchWidth
            [8, decodewidth, 8, decodewidth, 8, decodewidth],
            [12, decodewidth, 20, decodewidth, 16, decodewidth],
            [16, decodewidth, 32, decodewidth, 24, decodewidth],
            [24, decodewidth, 40, decodewidth, 32, decodewidth],
            [24, decodewidth, 40, decodewidth, 32, decodewidth]
            ]
    _isu_params = isu_params[decodewidth - 1]
    return _isu_params[0]+_isu_params[2]+_isu_params[4]
    
def estimate_bias_logic(feature,label):
    feature_list = [int(feature[item]+0.01) for item in range(feature.shape[0])]
    num_of_feature = len(set(feature_list))
    if num_of_feature<=2:
        logic_bias = 4
    else:
        reg = LinearRegression().fit(feature.reshape(feature.shape[0],1), label.reshape(label.shape[0],1))
        bias = reg.intercept_
        alpha = reg.coef_[0]
        logic_bias = bias / alpha     
    return
    
def estimate_bias_dtlb(feature,label):
    feature_list = [int(feature[item]+0.01) for item in range(feature.shape[0])]
    num_of_feature = len(set(feature_list))
    if num_of_feature<=1:
        dtlb_bias = 8
    else:
        reg = LinearRegression().fit(feature.reshape(feature.shape[0],1), label.reshape(label.shape[0],1))
        bias = reg.intercept_
        alpha = reg.coef_[0]
        dtlb_bias = bias / alpha      
    return
    
# transform the label for machine learning part with resource function
# input: component_name: which component is being processed, feature: feature related to this component, label: label of this component, which is to be transformed 
# return: transformed label
def encode_arch_knowledge(component_name,feature,label):
        
    if component_name=="BP" or component_name=="ICache" or component_name=="DCache" or component_name=="RNU" or component_name=="ROB" or component_name=="IFU" or component_name=="LSU":
        scale_factor = np.ones(label.shape)
        for i in range(len(encode_table[component_name])):
            encode_index = encode_table[component_name][i]
            acc_feature = feature[:,encode_index]
            scale_factor = scale_factor * acc_feature
        encode_label = label / scale_factor
    elif component_name=="Regfile":
        scale_factor = np.zeros(label.shape)
        for i in range(len(encode_table[component_name])):
            encode_index = encode_table[component_name][i]
            acc_feature = feature[:,encode_index]
            scale_factor = scale_factor + acc_feature
        encode_label = label / scale_factor
    elif component_name=="ISU":
        encode_index = encode_table[component_name][0]
        decodewidth = feature[:,encode_index]
        reserve_station = np.array([compute_reserve_station_entries(decodewidth[i]) for i in range(decodewidth.shape[0])])
        encode_label = label / reserve_station
    elif component_name=="OtherLogic":
        encode_index = encode_table[component_name][0]
        estimate_bias_logic(feature[:,encode_index],label)
        encode_label = label / (feature[:,encode_index] + logic_bias)
    elif component_name=="Dtlb":
        encode_index = encode_table[component_name][0]
        estimate_bias_dtlb(feature[:,encode_index],label)
        encode_label = label / (feature[:,encode_index] + dtlb_bias)
    else:
         encode_label = label / 1.0
    return encode_label

def train_model(mod, train_mod_feature, train_mod_label):
    mod.fit(train_mod_feature,train_mod_label)
    return mod

def build_model_for_one_component(component_name, select_option, build_feat, build_label):
    build_transformed_label = encode_arch_knowledge(component_name, build_feat, build_label)    
    if select_option == 0:
        model = xgb.XGBRegressor()
    else:
        model = GradientBoostingRegressor()
    trained_model = train_model(model,build_feat,build_transformed_label)
    return trained_model

def build_model(feature,label):
    model_list = []
    iter = 0
    for component in event_feature_of_components.keys():
        # get model option
        model_select_option = component_model_selection[iter]
            
        # get respective feature and label
        start_event = event_feature_of_components[component][0]
        end_event = event_feature_of_components[component][1]
        feature_index = params_feature_of_components[component] + [item for item in range(start_event,end_event)]
        component_feature = feature[:,feature_index]
        label_index = iter + 1
        component_label = label[:,label_index]
            
        # build model
        ml_model_this_component = build_model_for_one_component(component,model_select_option,component_feature,component_label)
        model_list.append(ml_model_this_component)
        iter = iter + 1
    return model_list

# compute the final power estimation
# input: component_name: which component's power is being computed, feature: feature is used to compute resource function, pred: the result of machine learning part
def decode_arch_knowledge(component_name,feature,pred):
    if component_name=="BP" or component_name=="ICache" or component_name=="DCache" or component_name=="RNU" or component_name=="ROB" or component_name=="IFU" or component_name=="LSU":
        scale_factor = np.ones(pred.shape)
        for i in range(len(encode_table[component_name])):
            decode_index = encode_table[component_name][i]
            acc_feature = feature[:,decode_index]
            scale_factor = scale_factor * acc_feature
        decode_pred = pred * scale_factor
    elif component_name=="Regfile":
        scale_factor = np.zeros(pred.shape)
        for i in range(len(encode_table[component_name])):
            decode_index = encode_table[component_name][i]
            acc_feature = feature[:,decode_index]
            scale_factor = scale_factor + acc_feature
        decode_pred = pred * scale_factor
    elif component_name=="ISU":
        decode_index = encode_table[component_name][0]
        decodewidth = feature[:,decode_index]
        reserve_station = np.array([compute_reserve_station_entries(decodewidth[i]) for i in range(decodewidth.shape[0])])
        decode_pred = pred * reserve_station
    elif component_name=="OtherLogic":
        decode_index = encode_table[component_name][0]
        decode_pred = pred * (feature[:,decode_index] + logic_bias)
    elif component_name=="Dtlb":
        decode_index = encode_table[component_name][0]
        decode_pred = pred * (feature[:,decode_index] + dtlb_bias)
    else:
        decode_pred = pred * 1.0
    return decode_pred
    
    
# compute power for one component
# input: component_name: which component is being processed, model: the machine learning model of this component, test_feat: the related feature of this component
# return: a power prediction for this component
def test_for_one_component(component_name, model, test_feat):
    pred_part = model.predict(test_feat)
    power_pred = decode_arch_knowledge(component_name,test_feat,pred_part)
    return power_pred
    

def predict_per_comp(model_list,test_feature):
    iter = 0
    power_value = np.zeros(test_feature.shape[0])
    power_list = []
    for component in event_feature_of_components.keys():
        # get model
        model_component = model_list[iter]
            
        # get respective feature and label
        start_event = event_feature_of_components[component][0]
        end_event = event_feature_of_components[component][1]
        feature_index = params_feature_of_components[component] + [item for item in range(start_event,end_event)]
        component_feature = test_feature[:,feature_index]
            
        # compute and accumulate power
        power_component = test_for_one_component(component,model_component,component_feature)
        power_list.append(power_component)
            
        iter = iter + 1
            
    return power_list

# test the PANDA
# input: test_feature: the feature of testing set, model_list: the machine learning part of PANDA
# return: total power prediction
def predict(model_list,test_feature):
    power_list = predict_per_comp(model_list,test_feature)
    power_value = np.zeros(test_feature.shape[0])
    for power in power_list:
        power_value = power_value + power      
    return power_value

def data_augment(source_model,target_feature,target_label,valid_index):
    #feature_augment = None
    #label_augment = None
    pseudo_label_list = predict_per_comp(source_model,target_feature)
    gt_feature = target_feature[valid_index]
    gt_label = target_label[valid_index]
    #print(len(pseudo_label_list))
    #print(pseudo_label_list[0].shape)
    #print(target_feature.shape,target_label.shape)
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
            #print(diff.shape)
            diff = np.linalg.norm(diff,axis=1)
            #print(diff.shape)
            similar_sample_idx = np.argmin(diff)
            #print(similar_sample_idx)
            calibrated_pseudo_label[sample_idx,iter+1] = pseudo_label_list[iter][sample_idx] * (gt_label[similar_sample_idx,iter+1] / pseudo_label_list[iter][valid_index[similar_sample_idx]])
            iter = iter + 1
            
        calibrated_pseudo_label[sample_idx,0] = np.sum(calibrated_pseudo_label[sample_idx,1:])
    return target_feature, calibrated_pseudo_label

def unknown_n_config(unknown,target_uarch,source_uarch):
    
    source_feature, source_label = load_data(source_uarch)
    target_feature, target_label = load_data(target_uarch)
        
    # print(source_feature.shape)
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
        
        source_model = build_model(source_feature, source_label)
        feature_augment, label_augment = data_augment(source_model,target_feature,target_label,training_set)
        target_model = build_model(feature_augment, label_augment)
        prediction = predict(target_model, target_feature[testing_set])

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

for i in range(1,10):
    r,mape = unknown_n_config(i,"XS","BOOM")
    curve_mape.append(mape)
    curve_r.append(r)

np.save("xs_transfer_curve_mape.npy",np.array(curve_mape))
np.save("xs_transfer_curve_r.npy",np.array(curve_r))
    
curve_mape = []
curve_r = []    

for i in range(1,15):
    r,mape = unknown_n_config(i,"BOOM","XS")
    curve_mape.append(mape)
    curve_r.append(r)
    
np.save("panda_transfer_curve_mape.npy",np.array(curve_mape))
np.save("panda_transfer_curve_r.npy",np.array(curve_r))