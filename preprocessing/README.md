## Data Preprocessing

### `ere`
1. Download ERE English data from LDC, specifically, 
- "LDC2015E29_DEFT_Rich_ERE_English_Training_Annotation_V2"
- "LDC2015E68_DEFT_Rich_ERE_English_Training_Annotation_R2_V2"
- "LDC2015E78_DEFT_Rich_ERE_Chinese_and_English_Parallel_Annotation_V2"

2. Collect all these data under `Dataset` with such setup:
```
Dataset
├── ERE_EN
│   ├── LDC2015E29_DEFT_Rich_ERE_English_Training_Annotation_V2
│   │     ├── data
│   │     ├── docs
│   │     └── ...
│   ├── LDC2015E68_DEFT_Rich_ERE_English_Training_Annotation_R2_V2
│   │     ├── data
│   │     ├── docs
│   │     └── ...
│   └── LDC2015E78_DEFT_Rich_ERE_Chinese_and_English_Parallel_Annotation_V2
│       ├── data
│       ├── docs
│       └── ...
└── ERE_ES
    └── LDC2015E107_DEFT_Rich_ERE_Spanish_Annotation_V2
        ├── data
        ├── docs
        └── ...
```

3. Run `./process_ere.sh`

### `ace`
1. Download ACE 2005 Data from [LDC](https://catalog.ldc.upenn.edu/LDC2006T06), you will get a folder called `ace_2005_td_v7`, put it under `Dataset` with such setup:
```
Dataset
└── ace_2005_td_v7
    ├── Arabic
    │     ├── bn
    │     ├── nw
    │     └── ...
    ├── Chinese
    │     ├── bn
    │     ├── nw
    │     └── ...
    └── English
        ├── bc
        ├── bn
        └── ...
```
We will use this for our `ace_zh` part data

2. For english data, we use the code from [Xu et al.](https://github.com/fe1ixxu/Gradual-Finetune/tree/master/dygiepp/data/ace-event/processed-data). Put their `processed-data` under `Dataset` with such setup:
```
Dataset
└── ace_2005_Xuetal
    ├── ar
        └── json
             └── ...

```

3. Run `./process_ace.sh`

4. For arabic data, we thanks the authors from the paper [Language Model Priming for Cross-Lingual Event Extraction](https://arxiv.org/abs/2109.12383) to share the data. They download data from [Xu et al.](https://github.com/fe1ixxu/Gradual-Finetune/tree/master/dygiepp/data/ace-event/processed-data) and perform clean up for arabic sentences. Their shared data is put at the folder `Dataset/ace_2005/ar`. We processed their split data and form our final data in `../processed_data/ace05_ar_mT5`
