# FirePower
FirePower: Towards a Foundation with Generalizable Knowledge for Architecture-Level Power Modeling

Power efficiency is a critical design objective in modern processor design. A high-fidelity architecture-level power modeling method is greatly needed by CPU architects for guiding early optimizations. However, traditional architecture-level power models can not meet the accuracy requirement, largely due to the discrepancy between the power model and actual design implementation. While some machine learning (ML)-based architecture-level power modeling methods have been proposed in recent years, the data-hungry ML model training process requires sufficient similar known designs, which are unrealistic in many development scenarios. 

This work proposes a new power modeling solution FirePower that targets the few-shot learning scenario for new target architectures. FirePower proposes multiple new policies to utilize cross-architecture knowledge. First, it develops power models at the component level, and components are defined in a power-friendly manner. Besides, it supports different generalization strategies for models of different components. Moreover, it formulates generalizable and architecture-specific design knowledge into two separate models. FirePower also supports the evaluation of the generalization quality for any target architecture. In our experiments, FirePower can achieve a low error percentage of 5.8\% and a high correlation $R$ of 0.98 on average only using two configurations of the target architecture. This is 8.8\% lower in error percentage and 0.03 higher in $R$ compared with directly training McPAT-Calib baseline on configurations of the target architecture.

## Quick Start
```
cd src/FirePower
python FirePower.py
```
