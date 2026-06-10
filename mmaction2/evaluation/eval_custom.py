import argparse
import copy
import json

import numpy as np

# from utils import match_temporal_segments
import utils

# TODO(refactor): Centralize label constants to a config management module.
from constants import LABEL_MAP, LABEL_NAME2ID, LabelMap

from typing import List

from .eval_detection import ActivityNetLocalization


# python -m mmaction2.evaluation.eval_custom -eg
class AdvancedDualEvaluator(ActivityNetLocalization):
    """Extended ActivityNetLocalization for dual-stage verification.

    1. Positioning (Where): Verifies temporal segment boundaries.
    2. Decision (What): Verifies Trigger precision and accuracy.
    """

    def __init__(
        self,
        ground_truth_filename=None,
        prediction_filename=None,
        verbose=False,
    ):
        """Initializes the evaluator with GT and prediction files."""
        self._label_map = LABEL_MAP
        self._label_name2id = LABEL_NAME2ID
        # Inherit base initialization (handles JSON reading and basic parsing).
        super().__init__(
            ground_truth_filename,
            prediction_filename,
            verbose=verbose,
        )
        # Backup original parsed data for dynamic mode switching.
        self._orig_ground_truth = copy.deepcopy(self.ground_truth)
        self._orig_prediction = copy.deepcopy(self.prediction)
        self._orig_activity_index = copy.deepcopy(self.activity_index)

    @staticmethod
    def _import_ground_truth(ground_truth_filename):
        """Imports ground truth data and filters out stable background."""
        with open(ground_truth_filename, "r") as f:
            data = json.load(f)
        ground_truth = []
        for video_id, video_info in data.items():
            if video_id == "version":
                continue
            for anno in video_info["annotations"]:
                label = int(anno["label"])
                # We do not evaluate "Stable" (Label 1) as it is background.
                if label == LABEL_NAME2ID[LabelMap.STABLE]:
                    continue
                ground_truth.append(
                    {
                        "video-id": video_id,
                        "t-start": float(anno["segment"][0]),
                        "t-end": float(anno["segment"][1]),
                        "label": label,
                    }
                )
        activity_index = LABEL_MAP
        return ground_truth, activity_index

    @staticmethod
    def _import_prediction(prediction_filename):
        """Imports prediction results for analysis."""
        with open(prediction_filename, "r") as f:
            data = json.load(f)
        prediction = []
        for video_id, video_info in data["results"].items():
            for result in video_info:
                label = int(result["label"])
                prediction.append(
                    {
                        "video-id": video_id,
                        "t-start": float(result["segment"][0]),
                        "t-end": float(result["segment"][1]),
                        "label": label,
                        "score": result.get("score", 1.0),
                    }
                )
        return prediction

    def _apply_mapping(self, data_list, map_all_to=0):
        """Forces all labels in the dataset to a specific ID for evaluation."""
        new_data = copy.deepcopy(data_list)
        for item in new_data:
            item["label"] = map_all_to
        return new_data

    def _apply_filter(self, data_list, targets: list = None):
        """Filters the dataset to retain only the target label."""
        if targets is None:
            raise ValueError("Targets must be provided.")
        return [
            copy.deepcopy(item)
            for item in data_list
            if int(item["label"]) in targets
        ]

    @staticmethod
    def _apply_safe_shift_temporal(data_list, l_shift=0.0, r_shift=0.1):
        """Safe temporal shift to avoid overlapping with the next event."""
        if not data_list:
            return data_list

        new_data = copy.deepcopy(data_list)
        # Must sort by video and start time to identify neighbors.
        new_data.sort(key=lambda x: (x["video-id"], x["t-start"]))

        for i in range(len(new_data)):
            curr_item = new_data[i]
            orig_end = curr_item["t-end"]
            shifted_end = orig_end + r_shift

            if i + 1 < len(new_data):
                next_item = new_data[i + 1]
                if curr_item["video-id"] == next_item["video-id"]:
                    # Prevent shifted end from eating into next start time.
                    shifted_end = min(
                        shifted_end,
                        next_item["t-start"] - 0.1,
                    )

            curr_start = curr_item["t-start"]
            curr_item["t-start"] = max(curr_start, orig_end - l_shift)
            curr_item["t-end"] = max(orig_end, shifted_end)

        return new_data

    def set_eval_mode(
        self,
        mode: str = "stage1",
        target_labels: List[int] = None,
        tiou_thresholds: List[float] = None,
        shifts: List[float] = None,
    ):
        """Dynamically switches between Positioning and Decision modes."""
        if tiou_thresholds is None:
            raise ValueError("tiou_thresholds must be provided.")
        self.tiou_thresholds = tiou_thresholds

        if mode == "stage1":  # Unstable Interval
            # Evaluate overall motion localization by mapping all to ID 0.
            self.ground_truth = self._apply_mapping(
                self._orig_ground_truth,
                map_all_to=0,
            )
            self.prediction = self._apply_mapping(
                self._orig_prediction,
                map_all_to=0,
            )
            self.activity_index = {"AnyMotion": 0}

        elif mode == "stage2":
            if shifts is None:
                raise ValueError("Shift value required for Stage 2.")

            if isinstance(shifts, list):
                if len(shifts) == 2:
                    l_shift, r_shift = shifts
                elif len(shifts) == 1:
                    l_shift, r_shift = 0, shifts
            else:
                raise ValueError("Shifts [l_shift, r_shift] must be provided.")

            self.ground_truth = self._apply_filter(
                self._orig_ground_truth,
                targets=target_labels,
            )
            self.prediction = self._apply_filter(
                self._orig_prediction,
                targets=target_labels,
            )

            self.ground_truth = self._apply_safe_shift_temporal(
                self.ground_truth,
                l_shift=l_shift,
                r_shift=r_shift,
            )
            self.prediction = self._apply_safe_shift_temporal(
                self.prediction,
                l_shift=0.0,  # Fix, 用來定位 Trigger point
                r_shift=r_shift,  # dummy
            )

            self.ground_truth = self._apply_mapping(
                self.ground_truth, map_all_to=0
            )
            self.prediction = self._apply_mapping(
                self.prediction, map_all_to=0
            )
            self.activity_index = dict(
                [
                    (f"Trigger_{target_label}", target_label)
                    for target_label in target_labels
                ]
            )

        print(
            f"Mode set: {mode.upper()} | "
            f"GT: {len(self.ground_truth)} | "
            f"PD: {len(self.prediction)}"
        )

    def evaluate_low_level_benchmark(self):
        gt_by_video = {}
        for item in self.ground_truth:
            gt_by_video.setdefault(item["video-id"], []).append(item)
        # gt_by_video = merge_custom_format_segments(gt_by_video, union_ms=500,)

        pd_by_video = {}
        for item in self.prediction:
            pd_by_video.setdefault(item["video-id"], []).append(item)

        all_vids = set(list(gt_by_video.keys()) + list(pd_by_video.keys()))
        video_reports = {}
        total_stats = {
            "matched": 0,
            "not_tight": 0,
            "gt_orphan": 0,
            "pd_orphan": 0,
        }
        for vid in sorted(all_vids):
            v_gts = sorted(
                gt_by_video.get(vid, []), key=lambda x: x["t-start"]
            )
            v_pds = sorted(
                pd_by_video.get(vid, []), key=lambda x: x["t-start"]
            )
            gt_segments = [[v_gt["t-start"], v_gt["t-end"]] for v_gt in v_gts]
            pd_segments = [[v_pd["t-start"], v_pd["t-end"]] for v_pd in v_pds]

            res = utils.match_temporal_segments(
                gt_segments,
                pd_segments,
                tiou_thres=0.5,
            )
            (matched, not_tight, gt_orphan, pd_orphan, iou_vals) = res

            # ==========================================
            # 3. 轉換統計指標 (Statistics Calculation)
            # ==========================================
            # 基礎計數
            cnt_matched = len(matched)
            cnt_not_tight = len(not_tight)
            cnt_gt_orphan = len(gt_orphan)
            cnt_pd_orphan = len(pd_orphan)

            # --- 嚴謹定義 (Strict Definition) ---
            # TP: 只有完美匹配才算分
            tp = cnt_matched
            # FN (Miss): 完全漏掉 + 抓了但不準 (Not Tight 對 GT 來說是 Miss)
            fn = cnt_gt_orphan + cnt_not_tight
            # FP (False Alarm): 亂報 + 報了但不準 (Not Tight 對 PD 來說是 False Alarm)
            fp = cnt_pd_orphan + cnt_not_tight

            # 計算 Precision / Recall / F1
            # 避免除以零
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            # ==========================================
            # 4. 輸出 Summary
            # ==========================================
            summary = {
                "meta": {
                    "total_gt": len(gt_segments),
                    "total_pd": len(pd_segments),
                    "iou_threshold": 0.5,
                },
                "raw_counts": {
                    "matched (Perfect Hit)": cnt_matched,
                    "not_tight (Localization Error)": cnt_not_tight,
                    "gt_orphan (Completely Missed)": cnt_gt_orphan,
                    "pd_orphan (Hallucination)": cnt_pd_orphan,
                },
                "metrics": {
                    "TP (True Positive)": tp,
                    "FN (False Negative / Undercount)": fn,
                    "FP (False Positive / False Alarm)": fp,
                    "Precision": round(precision, 4),
                    "Recall": round(recall, 4),
                    "F1-Score": round(f1_score, 4),
                },
            }

            total_stats["matched"] += cnt_matched
            total_stats["not_tight"] += cnt_not_tight
            total_stats["gt_orphan"] += cnt_gt_orphan
            total_stats["pd_orphan"] += cnt_pd_orphan
            video_reports[vid] = summary

        return {"summary": total_stats, "details": video_reports}

    def evaluate_trigger_benchmark(self):
        """Executes point-to-interval matching for trigger events."""
        gt_by_video = {}
        for item in self.ground_truth:
            gt_by_video.setdefault(item["video-id"], []).append(item)

        pd_by_video = {}
        for item in self.prediction:
            pd_by_video.setdefault(item["video-id"], []).append(item)

        all_vids = set(list(gt_by_video.keys()) + list(pd_by_video.keys()))
        video_reports = {}
        total_stats = {
            "just_fit": 0,
            "overcount": 0,
            "undercount": 0,
            "false_alarm": 0,
        }

        for vid in sorted(all_vids):
            v_gts = sorted(
                gt_by_video.get(vid, []), key=lambda x: x["t-start"]
            )
            v_pds = sorted(
                pd_by_video.get(vid, []), key=lambda x: x["t-start"]
            )
            gt_hit_counts = [0] * len(v_gts)
            pd_claimed = [False] * len(v_pds)

            for p_idx, pd in enumerate(v_pds):
                p_time = pd["t-start"]  # Start of PD as the trigger timestamp.
                candidates = [
                    g_idx
                    for g_idx, gt in enumerate(v_gts)
                    if gt["t-start"] <= p_time <= gt["t-end"]
                ]

                if candidates:
                    # Assign to GT with closest end-time for physical logic.
                    best_g_idx = min(
                        candidates,
                        key=lambda i: abs(v_gts[i]["t-end"] - p_time),
                    )
                    gt_hit_counts[best_g_idx] += 1
                    pd_claimed[p_idx] = True

            v_just_fit = sum(1 for c in gt_hit_counts if c == 1)
            v_overcount = sum(1 for c in gt_hit_counts if c > 1)
            v_undercount = sum(1 for c in gt_hit_counts if c == 0)
            v_false_alarm = pd_claimed.count(False)

            video_reports[vid] = {
                "just_fit": v_just_fit,
                "overcount": v_overcount,
                "undercount": v_undercount,
                "false_alarm": v_false_alarm,
                "total_pds": len(v_pds),
                "total_gts": len(v_gts),
                "miss_rate": (
                    v_undercount / len(v_gts) if len(v_gts) > 0 else 0
                ),
                "false_alarm_rate": (
                    v_false_alarm / len(v_pds) if len(v_pds) > 0 else 0
                ),
            }

            total_stats["just_fit"] += v_just_fit
            total_stats["overcount"] += v_overcount
            total_stats["undercount"] += v_undercount
            total_stats["false_alarm"] += v_false_alarm
            total_stats["miss_rate"] = None
            total_stats["false_alarm_rate"] = None

        return {"summary": total_stats, "details": video_reports}

    def generate_full_report(
        self,
        s1_tiou_thresholds=None,
        s2_tiou_thresholds=None,
        s2_shifts=None,
        verbose=False,
    ):
        """Generates the dual-stage HS-CODS evaluation report."""
        print("\n" + "=" * 80)
        print("🔬 HS-CODS Dual-Stage Evaluation Report")
        print("=" * 80)

        # Stage 1: Positioning evaluation.
        print("\n[Stage 1] Evaluating Low-Level Positioning...")
        self.set_eval_mode(
            mode="stage1",
            tiou_thresholds=s1_tiou_thresholds,
        )
        _, avg_mAP_s1 = self.evaluate()
        low_level_benchmark = self.evaluate_low_level_benchmark()
        s1_summary = low_level_benchmark["summary"]
        s1_details = low_level_benchmark["details"]

        # Generate Stage 1 report
        s1_metrics = self._low_level_report(
            s1_summary,
            s1_details,
            avg_mAP_s1,
            verbose=verbose,
        )

        # Stage 2: Trigger decision evaluation.
        print("\n[Stage 2] Evaluating Trigger Decision...")
        self.set_eval_mode(
            mode="stage2",
            target_labels=[
                LABEL_NAME2ID[LabelMap.TRIGGER],
                LABEL_NAME2ID[
                    LabelMap.NEGATIVE_UNSTABLE_EXTERNAL_DISTURBANCES
                ],
            ],
            tiou_thresholds=s2_tiou_thresholds,
            shifts=s2_shifts,
        )
        _, avg_mAP_s2 = self.evaluate()

        trigger_benchmark = self.evaluate_trigger_benchmark()
        s2_summary = trigger_benchmark["summary"]
        s2_details = trigger_benchmark["details"]

        # Generate Stage 2 report
        s2_metrics = self._trigger_report(
            s2_summary, s2_details, avg_mAP_s2, s2_shifts, verbose=verbose
        )

        # Combined Summary
        print("\n" + "=" * 80)
        print("📊 Combined Summary")
        print("=" * 80)
        header = (
            f"{'Metric':<30} | {'Stage 1 (Positioning)':<25} | "
            f"{'Stage 2 (Trigger)':<25}"
        )
        print(header)
        print("-" * 80)
        s1_prec = s1_metrics["precision"]
        s2_prec = s2_metrics["precision"]
        print(f"{'Precision':<30} | {s1_prec:>23.2%} | {s2_prec:>23.2%}")
        s1_rec = s1_metrics["recall"]
        s2_rec = s2_metrics["recall"]
        print(f"{'Recall':<30} | {s1_rec:>23.2%} | {s2_rec:>23.2%}")
        s1_f1 = s1_metrics["f1_score"]
        print(f"{'F1-Score':<30} | {s1_f1:>23.2%} | {'N/A':>25}")
        s2_pcr = s2_metrics["perfect_capture_rate"]
        print(f"{'Perfect Capture Rate':<30} | {'N/A':>25} | {s2_pcr:>23.2%}")
        s1_map = s1_metrics["mAP"]
        map_line = (
            f"{'mAP (ActivityNet)':<30} | {s1_map:>23.5f} | "
            f"{avg_mAP_s2:>23.5f}"
        )
        print(map_line)
        print("=" * 80 + "\n")

        return {
            "stage1": {
                "positioning_mAP": avg_mAP_s1,
                "precision": s1_metrics["precision"],
                "recall": s1_metrics["recall"],
                "f1_score": s1_metrics["f1_score"],
                "summary": s1_summary,
                "details": s1_details,
            },
            "stage2": {
                "decision_mAP": avg_mAP_s2,
                "precision": s2_metrics["precision"],
                "recall": s2_metrics["recall"],
                "perfect_capture_rate": s2_metrics["perfect_capture_rate"],
                "summary": s2_summary,
                "details": s2_details,
            },
        }

    def _low_level_report(
        self,
        summary,
        details,
        avg_mAP_s1,
        verbose=False,
    ):
        """Generates the low-level (Stage 1) positioning report."""
        # 1. Print Detailed Table
        print("\n" + "=" * 95)
        header = (
            f"{'VIDEO ID':<50} | {'MATCH':<6} | {'NOT_TIGHT':<9} | "
            f"{'GT_ORPHAN':<9} | {'PD_ORPHAN':<9}"
        )

        if verbose:
            print(header)
            print("-" * 95)
            for vid in sorted(details.keys()):
                v = details[vid]
                display_id = (vid[:47] + "...") if len(vid) > 50 else vid
                raw = v["raw_counts"]
                matched = raw["matched (Perfect Hit)"]
                not_tight = raw["not_tight (Localization Error)"]
                gt_orphan = raw["gt_orphan (Completely Missed)"]
                pd_orphan = raw["pd_orphan (Hallucination)"]
                print(
                    f"{display_id:<50} | {matched:6d} | "
                    f"{not_tight:9d} | {gt_orphan:9d} | {pd_orphan:9d}"
                )

            print(
                f"{'TOTAL SUMMARY':<50} | {summary['matched']:6d} | "
                f"{summary['not_tight']:9d} | {summary['gt_orphan']:9d} | "
                f"{summary['pd_orphan']:9d}"
            )
            print("=" * 95)

        # 2. Calculate aggregate metrics
        total_tp = summary["matched"]
        total_fn = summary["gt_orphan"] + summary["not_tight"]
        total_fp = summary["pd_orphan"] + summary["not_tight"]

        tp_plus_fp = total_tp + total_fp
        precision = total_tp / tp_plus_fp if tp_plus_fp > 0 else 0.0
        tp_plus_fn = total_tp + total_fn
        recall = total_tp / tp_plus_fn if tp_plus_fn > 0 else 0.0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # 3. Print Summary Report
        print("\n" + "=" * 60)
        print("📊 Stage 1: Low-Level Positioning Benchmark")
        print("=" * 60)
        print(f"  ● Matched (Perfect Hit)      : {summary['matched']:4d} | TP")
        print(
            f"  ● Not Tight (Localization Err): "
            f"{summary['not_tight']:4d} | Partial Match"
        )
        print(
            f"  ● GT Orphan (Completely Missed): "
            f"{summary['gt_orphan']:4d} | FN"
        )
        pd_orphan_val = summary["pd_orphan"]
        print(f"  ● PD Orphan (Hallucination)   : {pd_orphan_val:4d} | FP")
        print("-" * 60)
        print(f"  ◎ Precision (TP/(TP+FP))      : {precision:.2%}")
        print(f"  ◎ Recall (TP/(TP+FN))        : {recall:.2%}")
        print(f"  ◎ F1-Score                    : {f1_score:.2%}")
        print(f"  ◎ mAP (ActivityNet)          : {avg_mAP_s1:.5f}")
        print("=" * 60)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "mAP": avg_mAP_s1,
        }

    def _trigger_report(
        self,
        summary,
        details,
        avg_mAP_s2,
        s2_shifts,
        verbose=False,
    ):
        """Generates the trigger (Stage 2) decision report."""
        # 1. Print Detailed Table.

        if verbose:
            print("\n" + "=" * 95)
            header = (
                f"{'VIDEO ID':<50} | {'FIT':<5} | {'OVER':<5} | "
                f"{'UNDER':<5} | {'FA':<5} | "
                f"{'FA_R(%)':<5} | {'MISS_R(%)':<5} | "
            )
            print(header)
            print("-" * 95)
            for vid in sorted(details.keys()):
                v = details[vid]
                display_id = (vid[:47] + "...") if len(vid) > 50 else vid
                just_fit = v["just_fit"]
                overcount = v["overcount"]
                undercount = v["undercount"]
                false_alarm = v["false_alarm"]
                miss_rate = v["miss_rate"]
                false_alarm_rate = v["false_alarm_rate"]
                print(
                    f"{display_id:<50} | {just_fit:5d} | {overcount:5d} | "
                    f"{undercount:5d} | {false_alarm:5d} | "
                    f"{false_alarm_rate:3.2f} | {miss_rate:3.2f}"
                )

            total_line = (
                f"{'TOTAL SUMMARY':<50} | {summary['just_fit']:5d} | "
                f"{summary['overcount']:5d} | {summary['undercount']:5d} | "
                f"{summary['false_alarm']:5d}"
            )
            print(total_line)
            print("=" * 95)

        # 2. Business Metrics Calculation.
        total_actions = (
            summary["just_fit"] + summary["overcount"] + summary["undercount"]
        )
        denominator = (
            summary["just_fit"]
            + summary["overcount"]
            + summary["false_alarm"]
            + 1e-6
        )
        precision = summary["just_fit"] / denominator
        recall = (summary["just_fit"] + summary["overcount"]) / (
            total_actions + 1e-6
        )

        # 3. Print Summary Report
        print("\n" + "=" * 60)
        print("📊 Stage 2: Trigger Decision Benchmark")
        print("=" * 60)
        if isinstance(s2_shifts, list) and len(s2_shifts) == 2:
            shift_str = f"[{s2_shifts[0]}, {s2_shifts[1]}]"
        else:
            shift_str = str(s2_shifts)
        print(f"  Shift Configuration: {shift_str}s")
        just_fit_val = summary["just_fit"]
        print(f"  ● Just Fit  (1:1) : {just_fit_val:4d} | Perfect")
        overcount_val = summary["overcount"]
        print(f"  ● Overcount (>1) : {overcount_val:4d} | Multi-trigger")
        print(f"  ● Undercount (0)  : {summary['undercount']:4d} | Missed")
        # trigger 在 gt 區間以外: 沒有任何 gt 區間 收留到 Trigger 點.
        print(f"  ● False Alarm     : {summary['false_alarm']:4d} | Noise")
        print("-" * 60)
        # overcount 我們將視為失敗 (過度敏感) [多匡時會使這指標下降]
        perfect_rate = summary["just_fit"] / (total_actions + 1e-6)
        print(f"  ◎ Perfect Capture Rate : {perfect_rate:.2%}")
        # 正確觸發且沒有重複觸發 (v)
        print(f"  ◎ System Reliability (precision)  : {precision:.2%}")
        # 確認是否至少有辦法抓到 gt. 通常這裡突顯的是 false alarm.
        print(f"  ◎ Action Recall      (recall)     : {recall:.2%}")
        print(f"  ◎ mAP (ActivityNet) : {avg_mAP_s2:.5f}")
        print("=" * 60)

        return {
            "precision": precision,
            "recall": recall,
            "perfect_capture_rate": perfect_rate,
        }


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Report detection mAP for ActivityNet.",
    )
    parser.add_argument("--gt", type=str, help="Ground truth file path.")
    parser.add_argument("--pd", type=str, help="Prediction file path.")
    parser.add_argument("--config", type=str, default=None,
                        help="run.yaml; derives gt/pd from output.dir")
    parser.add_argument("-d", "--demo", action="store_true", help="Run demo.")
    parser.add_argument(
        "-eg", "--example", action="store_true", help="Show example."
    )
    return parser.parse_args()


# python -m mmaction2.evaluation.eval_custom -eg
def main():
    args = parse_args()
    OUTPUT_DIR = "Block_0119"
    # OUTPUT_DIR = "Block_TEST"
    TRIGGER_MODE = "yolo"
    # TRIGGER_MODE = "ema_phash"
    FOLDER = f"{OUTPUT_DIR}_{TRIGGER_MODE}"
    pd = (
        f"/Users/eason.hung/Documents/Projects/explus/output/{FOLDER}/"
        f"predictions/merge_data.json"
    )
    gt = (
        f"/Users/eason.hung/Documents/Projects/explus/output/{FOLDER}/"
        f"ground_truth/data.json"
    )
    if args.example:
        usage = (
            "\nUsage example:\npython -m mmaction2.evaluation.eval_custom "
            "--gt <GT> --pd <PD>"
        )
        print(usage)
        example_cmd = (
            f"\npython -m mmaction2.evaluation.eval_custom \\\n    "
            f"--gt {gt} \\\n   --pd {pd}"
        )
        print(example_cmd)
        return

    if args.demo:
        print("Demo mode is currently a placeholder.")
        return

    # Standard execution. --config derives gt/pd from output.dir; explicit
    # --gt/--pd take priority.
    gt_path, pd_path = args.gt, args.pd
    if args.config:
        from src.config.run_config import RunConfig
        cfg = RunConfig.from_yaml(args.config)
        d_gt, d_pd = cfg.eval_paths()
        gt_path = gt_path or d_gt
        pd_path = pd_path or d_pd

    evaluator = AdvancedDualEvaluator(gt_path, pd_path, verbose=False)
    results = evaluator.generate_full_report(
        s1_tiou_thresholds=np.linspace(0.5, 0.95, 10),
        s2_tiou_thresholds=np.array([0.5]),
        s2_shifts=[0.5, 2],  # 向前向後
        verbose=True,
    )


def merge_custom_format_segments(data, union_ms=300):
    """
    支援新格式的區間合併
    格式範例: {'video_id': [{'t-start': 1.2, 't-end': 2.6, 'label': 0}, ...]}
    """
    union_sec = union_ms / 1000.0
    print(data)
    for video_id, segments in data.items():
        if not segments:
            continue

        # 1. 根據 label 分組
        groups = {}
        for seg in segments:
            label = seg["label"]
            if label not in groups:
                groups[label] = []
            groups[label].append(seg)

        merged_video_segments = []

        # 2. 針對每個 label 進行時間排序與合併
        for label, items in groups.items():
            # 按 t-start 排序
            items.sort(key=lambda x: x["t-start"])

            curr = items[0].copy()
            for i in range(1, len(items)):
                nxt = items[i]

                # 計算間隙: 下一個開始 - 目前結束
                gap = nxt["t-start"] - curr["t-end"]

                if gap <= union_sec:
                    # 合併: 更新結束時間為最大值
                    curr["t-end"] = max(curr["t-end"], nxt["t-end"])
                else:
                    # 間隙過大，存入並開啟新區間
                    merged_video_segments.append(curr)
                    curr = nxt.copy()

            # 加入最後一個
            merged_video_segments.append(curr)

        # 3. 更新該影片的內容
        data[video_id] = merged_video_segments

    return data


if __name__ == "__main__":
    main()
