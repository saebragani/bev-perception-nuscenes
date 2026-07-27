import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import (
    add_center_dist,
    filter_eval_boxes,
    load_gt,
    load_prediction,
)
from nuscenes.eval.common.utils import center_distance
from nuscenes.eval.detection.algo import accumulate, calc_ap
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.utils import category_to_detection_name
from scipy.spatial.distance import cdist


def prepare_pred_boxes(nusc: NuScenes, result_file_path: str, cfg):
    pred_boxes, meta = load_prediction(result_path=result_file_path, max_boxes_per_sample=500, box_cls=DetectionBox)
    add_center_dist(nusc, pred_boxes)  # Mutates pred_boxes inplace

    filter_eval_boxes(  # Mutates pred_boxes inplace
        nusc=nusc,
        eval_boxes=pred_boxes,
        max_dist=cfg.class_range,
    )
    return pred_boxes


def box_to_df(gt_boxes: EvalBoxes, pred_boxes: EvalBoxes):
    rows = []
    for boxes in gt_boxes.boxes.values():
        for box in boxes:
            rows.append(
                [
                    "gt",
                    box.sample_token,
                    box.translation[0],
                    box.translation[1],
                    np.linalg.norm(box.ego_translation[:2]),
                    box.detection_name,
                    box.detection_score,
                ]
            )

    for boxes in pred_boxes.boxes.values():
        for box in boxes:
            rows.append(
                [
                    "pred",
                    box.sample_token,
                    box.translation[0],
                    box.translation[1],
                    np.linalg.norm(box.ego_translation[:2]),
                    box.detection_name,
                    box.detection_score,
                ]
            )

    binned_df = pd.DataFrame(
        rows, columns=["set", "sample_token", "translation_x", "translation_y", "range", "class", "detection_score"]
    )
    binned_df["bin"] = pd.cut(binned_df["range"], bins=[0, 20, 30, 40, 50], labels=["0-20", "20-30", "30-40", "40-50"])

    return binned_df


def add_eval_to_df(df, threshold):
    df["eval"] = None

    tp_pred, tp_gt = [], []
    for (sample_token, cls), group in df.groupby(["sample_token", "class"]):
        gts = group[group["set"] == "gt"]
        if len(gts) == 0:
            continue
        preds = group[group["set"] == "pred"].sort_values("detection_score", ascending=False)
        if len(preds) == 0:
            continue

        pred_idx = preds.index.to_numpy()
        gt_idx = gts.index.to_numpy()
        D = cdist(
            preds[["translation_x", "translation_y"]].to_numpy(), gts[["translation_x", "translation_y"]].to_numpy()
        )

        claimed = np.zeros(len(gt_idx), dtype=bool)
        for r in range(D.shape[0]):
            row = D[r].copy()
            row[claimed] = np.inf
            j = int(np.argmin(row))
            if row[j] < threshold:
                claimed[j] = True
                tp_pred.append(pred_idx[r])
                tp_gt.append(gt_idx[j])

    df.loc[tp_pred, "eval"] = "TP"
    df.loc[tp_gt, "eval"] = "TP"
    df.loc[df["eval"].isnull() & (df["set"] == "pred"), "eval"] = "FP"
    df.loc[df["eval"].isnull() & (df["set"] == "gt"), "eval"] = "FN"
    df.loc[tp_pred, "ann_token"] = df.loc[tp_gt, "ann_token"].to_numpy()

    return df


def get_recall_df(binned_df):
    rows = []
    for (class_name, bin_val), group in binned_df.groupby(by=["class", "bin"], observed=True):
        gt_df = group[group["set"] == "gt"]
        tp_count = gt_df[gt_df["eval"] == "TP"].shape[0]
        fn_count = gt_df[gt_df["eval"] == "FN"].shape[0]
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) else float("nan")

        rows.append([class_name, bin_val, recall])

    recall_df = pd.DataFrame(rows, columns=["class", "bin", "recall"])
    return recall_df


def get_binned_eval_boxes(eval_boxes, hi, lo):
    binned_boxes = EvalBoxes()
    for sample_token in eval_boxes.sample_tokens:
        boxes_ls = []
        for box in eval_boxes[sample_token]:
            if lo <= np.linalg.norm(box.ego_translation[:2]) < hi:
                boxes_ls.append(box)
        binned_boxes.add_boxes(sample_token, boxes_ls)

    return binned_boxes


def get_ap(gt_boxes, pred_boxes, bins, dist_thresholds, cfg):
    rows = []
    for lo, hi in bins:
        binned_gt_boxes = get_binned_eval_boxes(gt_boxes, hi, lo)
        binned_pred_boxes = get_binned_eval_boxes(pred_boxes, hi, lo)

        for class_name in cfg.class_names:
            npos = sum(b.detection_name == class_name for b in binned_gt_boxes.all)
            if npos == 0:
                rows.append([f"{lo}-{hi}", class_name, np.nan])
                continue
            ap_ls = []
            for dist_th in dist_thresholds:
                detection_metric_data = accumulate(
                    gt_boxes=binned_gt_boxes,
                    pred_boxes=binned_pred_boxes,
                    class_name=class_name,
                    dist_fcn=center_distance,
                    dist_th=dist_th,
                )
                aps = calc_ap(
                    md=detection_metric_data,
                    min_recall=0.1,
                    min_precision=0.1,
                )
                ap_ls.append(aps)
            rows.append([f"{lo}-{hi}", class_name, np.nanmean(ap_ls)])

    ap_df = pd.DataFrame(rows, columns=["bin", "class", "ap"])

    return ap_df


def add_condition_to_df(df, nusc):
    df["scene_description"] = df["sample_token"].apply(
        lambda x: nusc.get("scene", nusc.get("sample", x)["scene_token"])["description"].lower()
    )
    df["night"] = df["scene_description"].apply(lambda x: "night" in x)
    df["rain"] = df["scene_description"].apply(lambda x: "rain" in x)

    night_tokens = set(df.loc[df["night"] & ~df["rain"], "sample_token"])
    rain_tokens = set(df.loc[~df["night"] & df["rain"], "sample_token"])
    night_and_rain_tokens = set(df.loc[df["night"] & df["rain"], "sample_token"])
    clear_day_tokens = set(df.loc[~df["night"] & ~df["rain"], "sample_token"])

    return df, night_tokens, rain_tokens, night_and_rain_tokens, clear_day_tokens


def add_ann_token_to_df(df, nusc):
    lookup = {}
    for sample_token in df.loc[df["set"] == "gt", "sample_token"].unique():
        sample = nusc.get("sample", sample_token)
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            class_name = category_to_detection_name(ann["category_name"])
            if class_name is None:  # categories outside the 10 detection classes
                continue
            key = (sample_token, class_name, round(ann["translation"][0], 3), round(ann["translation"][1], 3))
            lookup[key] = ann_token

    df["ann_token"] = None
    ann_token_ls = []
    for _, row in df[df["set"] == "gt"].iterrows():
        key = (row["sample_token"], row["class"], round(row["translation_x"], 3), round(row["translation_y"], 3))
        ann_token_ls.append(lookup[key])
    df.loc[df["set"] == "gt", "ann_token"] = ann_token_ls

    return df


def get_eval_boxes_from_sample_tokens(eval_boxes, sample_tokens):
    boxes = EvalBoxes()
    for sample_token in sample_tokens:
        boxes.add_boxes(sample_token, list(eval_boxes[sample_token]))
    return boxes


def get_ap_for_conds(gt_boxes, pred_boxes, condition_names, condition_tokens, dist_thresholds, cfg):
    rows = []
    for cond, cond_tokens in zip(condition_names, condition_tokens):
        cond_gt_boxes = get_eval_boxes_from_sample_tokens(gt_boxes, cond_tokens)
        cond_pred_boxes = get_eval_boxes_from_sample_tokens(pred_boxes, cond_tokens)

        for class_name in cfg.class_names:
            npos = sum(b.detection_name == class_name for b in cond_gt_boxes.all)
            if npos == 0:
                rows.append([cond, class_name, np.nan, 0])
                continue
            ap_ls = []
            for dist_th in dist_thresholds:
                detection_metric_data = accumulate(
                    gt_boxes=cond_gt_boxes,
                    pred_boxes=cond_pred_boxes,
                    class_name=class_name,
                    dist_fcn=center_distance,
                    dist_th=dist_th,
                )
                aps = calc_ap(
                    md=detection_metric_data,
                    min_recall=0.1,
                    min_precision=0.1,
                )
                ap_ls.append(aps)
            rows.append([cond, class_name, np.nanmean(ap_ls), npos])

    cond_ap_df = pd.DataFrame(rows, columns=["condition", "class", "ap", "npos"])
    return cond_ap_df


def run_bin_metric_chain(
    nusc: NuScenes,
    gt_boxes: EvalBoxes,
    result_file_path: str,
    bins: list,
    threshold: float,
    dist_thresholds: list,
    cfg,
    dest_path,
):
    pred_boxes = prepare_pred_boxes(nusc, result_file_path, cfg)
    binned_df = box_to_df(gt_boxes, pred_boxes)
    binned_df = add_eval_to_df(binned_df, threshold)
    binned_df, night_tokens, rain_tokens, night_and_rain_tokens, clear_day_tokens = add_condition_to_df(binned_df, nusc)
    binned_df = add_ann_token_to_df(binned_df, nusc)
    binned_df.to_csv(dest_path / "binned_df.csv", index=False)
    recall_df = get_recall_df(binned_df)
    recall_df.to_csv(dest_path / "recall_df.csv", index=False)
    ap_df = get_ap(gt_boxes, pred_boxes, bins, dist_thresholds, cfg)
    ap_df.to_csv(dest_path / "ap_df.csv", index=False)

    cond_ap_df = get_ap_for_conds(
        gt_boxes=gt_boxes,
        pred_boxes=pred_boxes,
        condition_names=["night", "rain", "night_and_rain", "clear_day"],
        condition_tokens=[night_tokens, rain_tokens, night_and_rain_tokens, clear_day_tokens],
        dist_thresholds=dist_thresholds,
        cfg=cfg,
    )
    cond_ap_df.to_csv(dest_path / "cond_ap_df.csv", index=False)
