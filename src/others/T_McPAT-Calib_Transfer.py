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
            "OtherLogic":[14,101]
        }
        
# each list represents the configuration parameters related to each component
params_feature_of_components={
            "OtherLogic":[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
        }

component_model_selection=[1,0,0,0,0,0,0,1,1,0,1,0,1]

def load_data(uarch):
    feature = np.load('npy/{}_panda_feature.npy'.format(uarch))
    label = np.load('npy/{}_panda_label_fine_grained.npy'.format(uarch))
    label[:,1] = label[:,0]
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
    feature_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13] + [item for item in range(start_event,end_event)]
    component_feature = feature[:,feature_index]
    component_label = label
            
    # build model
    model = build_model_for_one_component(component_feature,component_label)

    return model
    
def predict(model,test_feature):
    # get respective feature and label
    start_event = 14
    end_event = 101
    feature_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13] + [item for item in range(start_event,end_event)]
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
        feature_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13] + [item for item in range(start_event,end_event)]
            
        diff = gt_feature[:,feature_index] - target_feature[sample_idx,feature_index]
        #print(diff.shape)
        diff = np.linalg.norm(diff,axis=1)
        similar_sample_idx = np.argmin(diff)
        calibrated_pseudo_label[sample_idx] = pseudo_label[sample_idx] * (gt_label[similar_sample_idx] / pseudo_label[valid_index[similar_sample_idx]])
        
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
        
        #res_model = get_resource_function(source_feature, source_label)
        #augmented_res_model = res_augment(res_model,target_feature,target_label,training_set)
        
        source_model = build_model(source_feature, source_label[:,0])
        feature_augment, label_augment = data_augment(source_model,target_feature,target_label[:,0],training_set)
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

for i in range(9,10):
    r,mape = unknown_n_config(i,"XS","BOOM")
    #unknown_n_config(i,"BOOM","XS")
    curve_mape.append(mape)
    curve_r.append(r)

# np.save("T_McPAT-Calib_XS_from_BOOM_mape.npy",np.array(curve_mape))
# np.save("T_McPAT-Calib_XS_from_BOOM_r.npy",np.array(curve_r))
    
curve_mape = []
curve_r = []    

for i in range(14,15):
    r,mape = unknown_n_config(i,"BOOM","XS")
    #unknown_n_config(i,"BOOM","XS")
    curve_mape.append(mape)
    curve_r.append(r)
    
# np.save("T_McPAT-Calib_BOOM_from_XS_mape.npy",np.array(curve_mape))
# np.save("T_McPAT-Calib_BOOM_from_XS_r.npy",np.array(curve_r))