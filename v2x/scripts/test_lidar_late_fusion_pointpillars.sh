DATA="../data/DAIR-V2X/cooperative-vehicle-infrastructure-test"
OUTPUT="../cache/vic-late-lidar-test"
rm -r $OUTPUT
rm -r ../cache/tmps
mkdir -p $OUTPUT/result
mkdir -p $OUTPUT/inf/lidar
mkdir -p $OUTPUT/veh/lidar

INFRA_MODEL_PATH="../configs/vic3d/late-fusion-pointcloud/pointpillars"
INFRA_CONFIG_NAME="trainval_config_i.py"
INFRA_MODEL_NAME="vic3d_latefusion_inf_pointpillars_596784ad6127866fcfb286301757c949.pth"

VEHICLE_MODEL_PATH="../configs/vic3d/late-fusion-pointcloud/pointpillars"
VEHICLE_CONFIG_NAME="trainval_config_v.py"
VEHICLE_MODEL_NAME="vic3d_latefusion_veh_pointpillars_a70fa05506bf3075583454f58b28177f.pth"

SPLIT_DATA_PATH="../data/split_datas/cooperative-split-data.json"

CUDA_VISIBLE_DEVICES=$1
FUSION_METHOD=$2
DELAY_K=$3
EXTEND_RANGE_START=$4
EXTEND_RANGE_END=$5
TIME_COMPENSATION=$6
python test.py \
  --input $DATA \
  --output $OUTPUT \
  --model $FUSION_METHOD \
  --dataset vic-async \
  --k $DELAY_K \
  --split test_A \
  --split-data-path $SPLIT_DATA_PATH \
  --inf-config-path $INFRA_MODEL_PATH/$INFRA_CONFIG_NAME \
  --inf-model-path $INFRA_MODEL_PATH/$INFRA_MODEL_NAME \
  --veh-config-path $VEHICLE_MODEL_PATH/$VEHICLE_CONFIG_NAME \
  --veh-model-path $VEHICLE_MODEL_PATH/${VEHICLE_MODEL_NAME} \
  --device ${CUDA_VISIBLE_DEVICES} \
  --pred-class car \
  --sensortype lidar \
  --extended-range $EXTEND_RANGE_START -39.68 -3 $EXTEND_RANGE_END 39.68 1 \
  --overwrite-cache \
  $TIME_COMPENSATION