import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import copy
from sklearn.metrics import r2_score
from sklearn.metrics import pairwise 
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def positive_relation(x,a,b):
    return a*x+b

class ResFuncModel:
    def __init__(self):
        self.option = 0
        self.config_index = []
        self.bias = 0
        self.R = 0
        
    def predict(self,feature):
        if self.option==0:
            result = 1
        elif self.option==1: # multiply
            result = np.ones(feature.shape[0])
            for param in self.config_index:
                result = result * feature[:,param]
            result = result + self.bias
        else: # accumulation
            result = np.zeros(feature.shape[0])
            for param in self.config_index:
                result = result + feature[:,param]
            result = result + self.bias
        return result
    
    def generate_combination(self,num_param):
        combination = [[]]
        for iter in range(num_param):
            new_comb = []
            for option in combination:
                new_comb.append(option+[0])
                new_comb.append(option+[1])
            combination = new_comb
        decode_comb = []
        for option in combination:
            cur_opt = []
            for param in range(len(option)):
                if option[param] == 1:
                    cur_opt.append(param)
            decode_comb.append(cur_opt)
        return decode_comb
    
    def fit_option(self,option,config_index,feature,label):
        if option == 1:
            x = np.ones(feature.shape[0])
            for param in config_index:
                x = x * feature[:,param]
        else:
            x = np.zeros(feature.shape[0])
            for param in config_index:
                x = x + feature[:,param]
        params,params_cov = curve_fit(positive_relation,x,label)
        pred = params[0]*x + params[1]
        R = np.corrcoef(label,pred)[1][0]
        return R, params[1]/params[0]
    
    def better(self,R1,comb1,R2,comb2):
        if len(comb1)==len(comb2):
            return R1>R2
        elif len(comb1)>len(comb2):
            return R1>R2+0.01
        else:
            return R1+0.01>R2
    
    def fit(self,feature,label,comp):
        combination = self.generate_combination(feature.shape[1])
        
        #if comp == "DCacheTagArray" or comp == "DCacheDataArray":
        #    combination = [[0,1]]
        #    best_option = 1
        #    cur_R, cur_bias = self.fit_option(best_option,combination[0],feature,label)
        #    
        #    self.option = best_option
        #    self.config_index = combination[0]
        #    self.bias = cur_bias
        #    self.R = cur_R
        #    
        #    return
        
        # if comp == "ICacheDataArray":
        #     combination = [[0]]
        #     best_option = 1
        #     cur_R, cur_bias = self.fit_option(best_option,combination[0],feature,label)
            
        #     self.option = best_option
        #     self.config_index = combination[0]
        #     self.bias = cur_bias
        #     self.R = cur_R
            
        #     return
        
        # if comp == "BPTAGE" or comp == "BPBTB" or comp == "BPOther":
        #     self.option = 1
        #     self.config_index = [0]
        #     self.bias = 0
        #     self.R = -1
            
        #     return
            
        if feature.shape[1]>5:
            combination = [[1]]
        #print(len(combination),combination)
        best_R = 0
        best_comb = []
        best_bias = 0
        best_option = 0
        for comb in combination:
            for option in range(1,3):
                cur_R, cur_bias = self.fit_option(option,comb,feature,label)
                #if cur_R>0 and cur_R>best_R:
                if self.better(cur_R,comb,best_R,best_comb):
                    best_R = cur_R
                    best_bias = cur_bias
                    best_comb = comb
                    best_option = option
        self.option = best_option
        self.config_index = best_comb
        self.bias = best_bias
        self.R = best_R
        return

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
            "BPTAGE":[0],
            "BPBTB":[0],
            "BPOther":[0],
            "ISUInt":[1,8,9],
            "ISUMem":[1,8,9],
            "ISUFp":[1,8,9]
        }

component_model_selection=[1,0,0,0,0,0,0,1,1,0,1,0,1]

def load_data(uarch):
    feature = np.load('npy/{}_panda_feature.npy'.format(uarch))
    label = np.load('npy/{}_panda_label_fine_grained.npy'.format(uarch))
    print(label.shape)
    area = label.reshape(label.shape[0]//8,8,23)
    area = np.average(area,axis=1)
    print(area.shape)
    
    # area = np.load('{}_fine_grained_area.npy'.format(uarch))
    # area = np.vstack([area[:,0],area[:,0]-np.sum(area[:,1:],axis=1),area[:,1:].T]).T
    # print(area.shape)
    
    return feature, label, area

def train_model(mod, train_mod_feature, train_mod_label,comp):
    if comp==" ":
        mod.fit(train_mod_feature,train_mod_label)
    else:
        mod.fit(train_mod_feature,train_mod_label,comp)
    return mod

def build_model_for_one_component(res_model, res_feat, build_feat, build_label):
    res_function = res_model.predict(res_feat)
    build_transformed_label = build_label / res_function
    
    model = xgb.XGBRegressor()
    trained_model = train_model(model,build_feat,build_transformed_label," ")
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
    return np.vstack([power_value] + power_list).T

def get_resource_function(source_feature, source_label):
    res_model = []
    iter = 0
    
    for component in event_feature_of_components.keys():
        feature_index = params_feature_of_components[component]
        feature = source_feature[:,feature_index]
        feature = feature.reshape((source_feature.shape[0]//8,8,feature.shape[1]))[:,0,:]
        
        label_index = iter + 1
        label = source_label[:,label_index]
        
        model = ResFuncModel()
        if component!="Itlb":
            res_model_this_component = train_model(model,feature,label,component)
        print(component,model.option,model.config_index,model.bias,model.R)
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
    plt.savefig("T_simple/{}.jpg".format(name),dpi=200)
    plt.close()
    return

def unknown_n_config(unknown,target_uarch,source_uarch):
    
    source_feature, source_label, source_area = load_data(source_uarch)
    target_feature, target_label, target_area = load_data(target_uarch)
    
    num_of_workload = 8
    num_of_config = target_feature.shape[0] // num_of_workload
    
    fold = num_of_config
    test_size = num_of_workload * unknown
    pred_acc_vector = np.zeros((num_of_config*num_of_workload,23))
    
    res_model = get_resource_function(source_feature, source_area)
        
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
            
        target_model = build_model(target_feature[training_set], target_label[training_set], res_model)
        prediction = predict(target_model, res_model, target_feature[testing_set])
        #print(prediction.shape)
        pred_acc_vector[testing_set] = pred_acc_vector[testing_set] + prediction

    pred_acc_vector = pred_acc_vector / unknown
    for i in range(23):
        draw_figure(target_label[:,i],pred_acc_vector[:,i],figure_name[i]+"_{}_from_{}".format(target_uarch,source_uarch))
    label = target_label[:,0]
        
    #r_report = np.corrcoef(label,pred_acc_vector)[1][0]
    #mape_report = mean_absolute_percentage_error(label,pred_acc_vector)
    #print("Unknown_{}_config".format(unknown))
    #print("R = {}".format(r_report))
    #print("MAPE = {}%".format(mape_report * 100))
    return #r_report, mape_report


curve_mape = []
curve_r = []

for i in range(9,10):
    unknown_n_config(i,"XS","BOOM")
    #unknown_n_config(i,"BOOM","XS")
    #curve_mape.append(mape)
    #curve_r.append(r)

# np.save("T_XS_from_BOOM_mape.npy",np.array(curve_mape))
# np.save("T_XS_from_BOOM_r.npy",np.array(curve_r))
    
curve_mape = []
curve_r = []    

for i in range(14,15):
    unknown_n_config(i,"BOOM","XS")
    #unknown_n_config(i,"BOOM","XS")
    #curve_mape.append(mape)
    #curve_r.append(r)
    
# np.save("T_BOOM_from_XS_mape.npy",np.array(curve_mape))
# np.save("T_BOOM_from_XS_r.npy",np.array(curve_r))