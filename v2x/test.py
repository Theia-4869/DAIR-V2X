import sys
import os
import os.path as osp

sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

import argparse
import logging

logger = logging.getLogger(__name__)

from tqdm import tqdm
import numpy as np

from v2x_utils import range2box, id_to_str
from config import add_arguments
from dataset import SUPPROTED_DATASETS
from dataset.dataset_utils import save_pkl, save_json
from models import SUPPROTED_MODELS
from models.model_utils import Channel


def test_vic(args, dataset, model):
    idx = -1
    for VICFrame, filt in tqdm(dataset):
        idx += 1
        try:
            veh_id = dataset.data[idx][0]["vehicle_pointcloud_path"].split("/")[-1].replace(".pcd", "")
        except Exception:
            veh_id = VICFrame["vehicle_pointcloud_path"].split("/")[-1].replace(".pcd", "")

        pred = model(
            VICFrame,
            filt,
            None if not hasattr(dataset, "prev_inf_frame") else dataset.prev_inf_frame,
        )

        pred["veh_id"] = veh_id
        pred_dict = {
            "boxes_3d": pred["boxes_3d"].tolist(),
            "labels_3d": pred["labels_3d"].tolist(),
            "scores_3d": pred["scores_3d"].tolist(),
            "ab_cost": pipe.current_bytes(),
        }
        pipe.flush()
        not_car_index_list = []
        for i in range(len(pred_dict["labels_3d"])):
            if pred_dict["labels_3d"][i] != 2:
                not_car_index_list.append(i)
        not_car_index_list.reverse()
        for index in not_car_index_list:
                pred_dict["boxes_3d"].pop(index)
                pred_dict["labels_3d"].pop(index)
                pred_dict["scores_3d"].pop(index)
        
        save_json(pred_dict, osp.join(args.output, "result", pred["veh_id"] + ".json"))
        # save_pkl(pred, osp.join(args.output, "result", pred["veh_id"] + ".pkl"))


def test_single(args, dataset, model):
    for frame, filt in tqdm(dataset):
        pred = model(frame, filt)
        save_pkl(pred, osp.join(args.output, "result", frame.id["camera"] + ".pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    add_arguments(parser)
    args, _ = parser.parse_known_args()
    # add model-specific arguments
    SUPPROTED_MODELS[args.model].add_arguments(parser)
    args = parser.parse_args()

    if args.quiet:
        level = logging.ERROR
    elif args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
    )

    extended_range = range2box(np.array(args.extended_range))
    logger.info("loading dataset")

    dataset = SUPPROTED_DATASETS[args.dataset](
        args.input,
        args,
        split=args.split,
        sensortype=args.sensortype,
        extended_range=extended_range,
    )

    logger.info("loading model")
    if args.eval_single:
        model = SUPPROTED_MODELS[args.model](args)
        test_single(args, dataset, model)
    else:
        pipe = Channel()
        model = SUPPROTED_MODELS[args.model](args, pipe)
        test_vic(args, dataset, model)
