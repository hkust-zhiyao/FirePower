figure_name = ["Total", "OtherLogic",
"RNU", "Itlb", "Dtlb", "Regfile", "ROB", "IFU", "LSU", "FU_Pool", 
"ICacheTagArray", "ICacheDataArray", "ICacheOther",
"DCacheTagArray", "DCacheDataArray", "DCacheMSHR", "DCacheOther",
"BPTAGE", "BPBTB", "BPOther",
"ISUInt", "ISUMem", "ISUFp"]

config_name = ["FetchWidth","DecodeWidth","FetchBufferEntry","RobEntry","IntPhyRegister","FpPhyRegister","LDQ/STQEntry","BranchCount","MemIssue/FpIssueWidth","IntIssueWidth","DCache/ICacheWay","DTLBEntry","DCacheMSHR","ICacheFetchBytes"]

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
            "ICacheTagArray":[0,10],
            "ICacheDataArray":[0,10],
            "ICacheOther":[0,10],
            "DCacheTagArray":[8,10,11,12],
            "DCacheDataArray":[8,10,11,12],
            "DCacheMSHR":[8,12],
            "DCacheOther":[8,10,11,12],
            "BPTAGE":[0,7],
            "BPBTB":[0,7],
            "BPOther":[0,7],
            "ISUInt":[1,9],
            "ISUMem":[1,8],
            "ISUFp":[1,8]
        }