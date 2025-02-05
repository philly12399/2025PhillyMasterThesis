# Thesis ReadME Docker
## Data & Procedure
**Data**
* D1. Pointcloud pcd (kitti format) 
* D2. Detection bbox
* D3. gt_db， split point cloud of each detection bbox
* D4. dense_pcd， dense point cloud from point cloud reconstruction
* D5. NDT voxelize cache of detection
* D6. Tracking results
* D6*. Refined Tracking results
* D7. Ground Truth
* D8. Tracking Metrics

**Procedure**
* P1. Generate gt_db， `philly_utils/utils/create_gt_db_kitti.py`， D1+D2->D3 
* P2. Generate dense_pcd，`Point-MAE/` point cloud reconstruct, D3->D4
* P3. `Anti_Occlusion_Tracker/NDT_precalculate.py` ,pre-calculate NDT voxelize for detection ， D4->D5
* P4. Tracking,`Anti_Occlusion_Tracker/main.py`, D2+D4->D6
* P5. Post process tracking results by track-level confidence, `Anti_Occlusion_Tracker/post_process.sh`, refine D6->D6*
* P6. Evaluation,`Anti_Occlusion_Tracker/scripts/KITTI/evaluate.py` D6+D7->D8
* P7. Tracking Visualizer,`3D-Detection-Tracking-Viewer/tracking_viewer.py` D1+bbox label(D2/D6/D7)->visualize

## Installation
### Repos
```
mkdir thesis
cd thesis
git clone git@github.com:philly12399/Anti_Occlusion_Tracker.git
git clone git@github.com:philly12399/Point-MAE.git
git clone git@github.com:philly12399/philly_utils.git
git clone git@github.com:philly12399/3D-Detection-Tracking-Viewer.git
```
get ${thesis_dir}
### Data
${data_dir}: `nas/archive/MasterThesis/2025/philly/data`
`data/KITTI_tracking/training/velodyne`(D1): 20 kitti sequences (0~20) + 1 Wayside pingtung sequence (0021)
`data/KITTI_tracking/training/gt_det_set_seq`(D2): gtdet, 4 difficulty
`data/KITTI_tracking/training/merged_det`(D2): real detections

`data/point_mae/gt_db`(D3): output dir of P1, input dir of P2
`data/point_mae/dense_db`(D4): output dir of P2, input dir of P3
`data/AOT/NDT_pkl`(D5): output dir of P3, input dir of P4
`data/AOT/track_exp`(D6): output dir of P4, input dir of P5 or P6 
`data/AOT/track_exp`(D6*): output dir of P5, input dir of P6 
`data/KITTI_tracking/training/label_02`(D7): input dir of P6 
`data/AOT/track_exp`(D8): output dir of P6

### Docker
`nas/archive/MasterThesis/2025/philly/docker/AOT_20250205.tar`

```
cd nas/archive/MasterThesis/2025/philly/docker/
cat AOT_20250205.tar  | sudo docker import - aot

docker run -it --name aot --shm-size="16g"  -v {thesis_dir}/thesis/:/thesis \
-v {data_dir}:/mydata -v /tmp/.X11-unix:/tmp/.X11-unix  --runtime=nvidia \
--gpus=all aot:new bash 

cd thesis/
ln -s /mydata data_root #link for relative data path
```

## Demo
**P1**
```
conda activate AOT
cd /thesis/philly_utils/utils/
bash demo.sh
```
### Point-MAE 

**P2**
```
conda activate base
cd /thesis/Point-MAE
bash vis_demo.sh
```

### AOT
**P3,P4,P5,P6**
```
conda activate AOT
cd /thesis/Anti_Occlusion_Tracker
source env_docker.sh
bash NDT_precalculate.sh #P3
bash test.sh #P4
# gtdet no need to do P5, only det need
bash post_process.sh #P5
bash eval.sh #P6
```
NOTE: P4 read detections from `thesis/Anti_Occlusion_Tracker/data/KITTI/detection/{det_name}`
## Configs
### P1
`philly_utils/utils/create_gt_db_kitti.py`
args:
-k  "kitti data path"
-l  "label path"
-o  "output path"
COMBINATION=A1/A2/A3 (quick setting for [EXP](https://hackmd.io/sLvCpKfdQOKPOb_JWleWUA?both=&stext=9643%3A22%3A0%3A1738697866%3A7wbAlo))

![image](https://hackmd.io/_uploads/r134nJltyl.png)

OCC_FILTER=[-1,0,1,2,3] 
CLASS_FILTER=['car','cyclist']


### P2
config dirs:
`Point-MAE/cfgs/`
`Point-MAE/cfgs/dataset_configs/`

config explanation:
`Point-MAE/cfgs/pretrain_demo.yaml`
```
optimizer : {
  type: AdamW,
  kwargs: {
  lr : 0.001,
  weight_decay : 0.05
}}

scheduler: {
  type: CosLR,
  kwargs: {
    epochs: 300,
    initial_epochs : 10
}}
# dataset_configs/
dataset : {
  test : { _base_: cfgs/dataset_configs/Wayside-DEMO.yaml,
            others: {subset: 'test', npoints: 1024}}
}
# Point-MAE params
model : {
  NAME: Point_MAE,
  group_size: 32,
  num_group: 128,
  loss: cdl2,
  transformer_config: {
    mask_ratio: 0.9,
    mask_type: 'rand',
    trans_dim: 384,
    encoder_dims: 384,
    depth: 12,
    drop_path_rate: 0.1,
    num_heads: 6,
    decoder_depth: 4,
    decoder_num_heads: 6,
  },
  }

npoints: 1024
total_bs : 128
step_per_update : 1
max_epoch : 300

save_vis_txt: False
additional_cfg : {
  VOXEL_SIZE: 0.1,
  TARGET_PATH: './data/dense_db/demo', #dense_db output path
  VIS_NUM: -1, 
  START_INDEX: 0,
  REFLECT_AUG: True,
  ALIGN_XY: True,
  CONF_THRES: 0.0,
}
```
`Point-MAE/cfgs/dataset_configs/Wayside-DEMO.yaml`
```
NAME: AOT
N_POINTS: 1024
PCD_PATH: data/gt_db/demo/gt_database  #gt_db input path
INFO_PATH: data/gt_db/demo/info.pkl  #gt_db info input path
SEQ: ['0021'] #list of sequence id you want to run
```
### P3
config dirs:
`Anti_Occlusion_Tracker/configs/NDT/`
config explanation:
`Anti_Occlusion_Tracker/configs/NDT/NDT_precalculate_demo.yml`
```
##ONLY FOR NDT PREPROCESS
# ------------------- General Options -------------------------
description                  : AB3DMOT
seed                         : 0
# --------------- main.py
dataset                      : KITTI      
split                        : val        
det_name                     : diff3  # name of the detection 
cat_list                     : ['car','cyclist'] #['Car', 'Pedestrian', 'Cyclist']
label_format                 : KITTI     
# ------------------NDT    #NDT voxel settings
NDT_flag                     : True   
pcd_db_root                  : "./data/KITTI/dense_db/demo/" #dense_db input path
NDT_cfg                      : {'voxel_size': 0.5, 'overlap': True, 'min_pts_voxel': 5, 'noise': 0.05}
NDT_cache_root               : "./data/KITTI/NDT_pkl/det/" #NDT cache dir path
NDT_cache_name               : "demo_cache" #path for load/write    #NDT cache exp name
#---------------
seq_eval                     : [21]  #list of sequence id you want to run
```
### P4
config dirs:
`Anti_Occlusion_Tracker/configs/`
config explanation:
`Anti_Occlusion_Tracker/configs/KITTI_demo.yml`
```
##TRACKING CONFIG
# ------------------- General Options -------------------------
description                  : AB3DMOT
seed                         : 0
# --------------- main.py
save_root                    : ./data/track_exp/demo
dataset                      : KITTI      
split                        : val        
det_name                     : diff3  # name of the detection 
cat_list                     : ['car','cyclist'] #['Car', 'Pedestrian', 'Cyclist']
#--------------- SEQ
seq_eval                     : [21]  #list of sequence id you want to track
# --------------- model.py
ego_com                      : True     
#---------------- addition
label_format                 : wayside      # Kitti, wayside
label_coord                  : lidar      # lidar or camera coord
output_kf_cls                : True ## if true output kf car as KF_Car class
output_mode                  : 'kf' ## output by interpolate or kf predict
kf_initial_speed             : 30 #km/h
conf_thres                   : 0.0 #confidence threshold of detection
# ------------------NDT #NDT settings
NDT_flag                     : True # use NDT or not
pcd_db_root                  : "./data/KITTI/dense_db/demo/"    # dense_db input path
NDT_cfg                      : {'voxel_size': 0.5, 'overlap': True, 'min_pts_voxel': 5, 'noise': 0.05}
NDT_cache_root               : "./data/KITTI/NDT_pkl/det/" #NDT cache dir path
NDT_cache_name               : "demo_cache" #NDT cache exp name
NDT_thres                    : {'car': {'NDT_score': -15000,'max_dist': 6.0,'max_angle': 20},
                                'cyclist': {'NDT_score': -15000,'max_dist': 4.5,'max_angle': 20}}  
#---------------AB3DMOT base #AB3DMOT settings
base_param                   : {'car':{'algm': 'hungar', 'metric': 'giou_3d', 'thres': 0.2, 'min_hits': 5, 'max_age': 40},
                                'cyclist':{'algm': 'hungar', 'metric': 'dist_3d', 'thres': 2.0, 'min_hits': 5, 'max_age': 40}}          
# --------------- DA stage2 threshold
two_stage                   : True  # use 2 stage DA or not  
stage2_param                : {'car':{'algm': 'hungar', 'metric': 'dist_3d', 'thres': 3.0},
                                'cyclist':{'algm': 'hungar', 'metric': 'dist_3d', 'thres': 3.0}}    
```
NOTE: P4 read detections from `thesis/Anti_Occlusion_Tracker/data/KITTI/detection/{det_name}`
### P5
`Anti_Occlusion_Tracker/scripts/post_processing/trk_conf_threshold.py`
args: 
-e "dir of tracking exp result without post processing"
Confidence threshold:` thres_dict={'Car':3.240738, 'Cyclist':3.645319} `

### P6
config dirs:
`Anti_Occlusion_Tracker/configs/eval/`
config explanation:
`Anti_Occlusion_Tracker/configs/eval/KITTI_demo.yml`
```
##EVALUATION CONFIG
# ------------------- Dataset Name -------------------------
description                  : KITTI
# ------------------- Path Options -------------------------
gt_path                      : "./label/" 
calib_path                   : "../../data/KITTI/tracking/training/calib/" 
trk_path                     : "../../data/track_exp/demo/label" #tracking results path
out_path                     : "../../data/track_exp/demo/"  #evaluation path results 
exp_name                     : eval
# ------------------- Eval Options -------------------------
iou                          : 3D
threshold                    : [0.25,0.5]
class_name                   : ['car','cyclist']
gt_format                    : Wayside
trk_format                   : Wayside
num_hypo                     : 1
max_occlusion                : 4
max_truncation               : 2 #0,1,2
eval_seq                     : [21]
fov_filter                   : False
```
### P7
`3D-Detection-Tracking-Viewer/tracking_viewer.py`
args: 
-b (Kitti/Philly) Kitti or Philly coordinate
wayside label is under Philly coordinate

## Experiment-combination

| | P1 |P2| P3 | P4 | P5 | P6 |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| gtdet x wayside     | A1    | pretrain_demo.yaml     | NDT_precalculate_demo.yml     | KITTI_demo.yml     | X    |KITTI_demo.yml |
| gtdet x kitti  | A2     | pretrain_test_on_kitti_gtdet.yaml     | NDT_precalculate_gtdet.yml     | KITTI_gtdet_car.yml / KITTI_gtdet_cyclist.yml   | X     | KITTI_gtdet_car.yml / KITTI_gtdet_cyclist.yml      
| det x kitti car     | A3    |pretrain_test_on_kitti_det.yaml     | NDT_precalculate_det.yml    | KITTI_det_car.yml     | V | KITTI_det_car.yml    |

P1 params combination: (`/thesis/philly_utils/utils/create_gt_db_kitti.py`)
1.
![image](https://hackmd.io/_uploads/HJK8A1xFke.png)

2.
    A1,A2:` -l ../data/KITTI_tracking/training/gt_det_set_seq/diff3 `
    A3: `-l ../data/KITTI_tracking/training/merged_det/mpoint_rcnn`
NOTE: A3 should change `CLASS_FILTER` to `['car']`

NOTE: you should change or delete this section if you use other dataset ![image](https://hackmd.io/_uploads/r1otA1xtyl.png)

