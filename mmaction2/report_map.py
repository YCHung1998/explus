# Copyright (c) OpenMMLab. All rights reserved.
import os
import json
import argparse
from pathlib import Path

# import mmengine
import numpy as np

from evaluation.eval_detection import ActivityNetLocalization

args = None
# python mmaction2/report_map.py -eg
OUTPUT_DIR = 'output/dummy'
# OUTPUT_DIR = 'output/BlockBased'
OUTPUT_DIR = 'output/BlockBased_1230'
# OUTPUT_DIR = 'output/AdaptiveBlockBased'
# OUTPUT_DIR = 'output/PixelBased'
# OUTPUT_DIR = 'output/PixelBased_bdry'
# OUTPUT_DIR = 'output/PixelBased_bdry_v2'
# OUTPUT_DIR = 'output/PixelBased_bdry_1231_revised'
OUTPUT_DIR = 'output/PixelBased_bdry_1231'

# OUTPUT_DIR = 'output/PixelBased_bdry_1230_v2'

# OUTPUT_DIR = 'output/BlockBased_dct'
# OUTPUT_DIR = 'output/BlockBased_bdry'

# OUTPUT_DIR = 'output/phash'
# OUTPUT_DIR = 'output/phash_bdry'
# OUTPUT_DIR = 'output/phash_bdry_v2'

# OUTPUT_DIR = 'output_Grid_v1_revised'
# OUTPUT_DIR = 'output/yolo'

# OUTPUT_DIR = 'output/Grid_v1_revised'
# OUTPUT_DIR = 'output/Grid_v1'
# OUTPUT_DIR = 'output/Grid_v1_1230'
# OUTPUT_DIR = 'output_Grid_max'
# OUTPUT_DIR = 'output_Grid_mstd'


def compare_gt_pd_files(gt_json_path, pd_json_path):
    try:
        # 讀取 GT 檔案
        with open(gt_json_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        # 讀取 PD 檔案 (集中式格式)
        with open(pd_json_path, 'r', encoding='utf-8') as f:
            pd_data = json.load(f).get("results", {})

        total_gt = 0
        total_pd = 0
        filenames = sorted(gt_data.keys())

        print(f"{'No.':<3} | {'Filename':<45} | {'GT':<5} | {'PD':<5}")
        print("-" * 65)

        for i, fname in enumerate(filenames, 1):
            # GT 數量取自 annotations
            gt_count = len(gt_data[fname].get("annotations", []))
            # PD 數量取自 results 裡的對應 key
            pd_count = len(pd_data.get(fname, []))

            total_gt += gt_count
            total_pd += pd_count

            print(f"{i:<3} | {fname:<45} | {gt_count:<5} | {pd_count:<5}")

        print("-" * 65)
        print(f"Summary: {len(filenames)} files")
        print(f"Total -> GT: {total_gt} | PD: {total_pd}")

    except Exception as e:
        print(f"Error: {e}")


def stat_centralized_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        results = data.get("results", {})
        total_predict_count = 0
        file_list = list(results.keys())
        for i, filename in enumerate(file_list, 1):
            # 取得該 key 對應的 segment 數量
            count = len(results[filename])
            total_predict_count += count
            print(f"{i}. filename: {filename}, predict_count: {count}")
        print("-" * 30)
        print(f"{len(file_list)} files, total : {total_predict_count} predict_count")
    except Exception as e:
        print(f"讀取檔案失敗: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description='Report detection mAP for'
                                     'ActivityNet proposal file')
    parser.add_argument('--proposal', type=str, help='proposal file')
    parser.add_argument(
        '--gt',
        type=str,
        default='/Users/eason.hung/Documents/Projects/test-something/mmaction2/demo/ground_truth_equal_demo1.json',
        help='groundtruth file')
    parser.add_argument(
        '--cls',
        type=str,
        default='cuhk17_top1',
        choices=['cuhk17_top1'],
        help='the way to assign label for each '
        'proposal')
    parser.add_argument(
        '--pd',
        type=str,
        default='/Users/eason.hung/Documents/Projects/test-something/mmaction2/demo/prediction_equal_demo1.json',
        help='the path to store detection results')
    parser.add_argument(
        '-d',
        '--demo',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '-eg',
        '--example',
        action='store_true',
        default=False,
    )
    args = parser.parse_args()
    return args


def main():
    global args, cls_funcs
    args = parse_args()
    print(args)
    if args.example:
        print(f"""
python mmaction2/report_map.py \\
    --gt /Users/eason.hung/Documents/Projects/test-something/{OUTPUT_DIR}/ground_truth/data.json \\
    --pd /Users/eason.hung/Documents/Projects/test-something/{OUTPUT_DIR}/predictions/merge_data.json
    """)
        exit()
    # demo process
    if args.demo:
        script_dir = Path(__file__).resolve().parent
        gt_json_fns = sorted(list(script_dir.glob('./demo/ground_*.json')))
        pd_json_fns = sorted(list(script_dir.glob('./demo/prediction*.json')))

        print('This is demo mode.')
        for _gt, _pd in zip(gt_json_fns, pd_json_fns):
            print(f'Processing GT: {_gt}, PD: {_pd}')
            anet_detection = ActivityNetLocalization(
                str(_gt),
                str(_pd),
                tiou_thresholds=np.linspace(0.5, 0.95, 10),
                # tiou_thresholds=np.arange(0.2, 0.5, 0.05),
                verbose=True
            )
            mAP, average_mAP = anet_detection.evaluate()
            print('[RESULTS] Performance on ActivityNet detection task.\n'
                  f'mAP: {mAP}\nAverage-mAP: {average_mAP}\n')

    # normal process
    if not args.demo:
        anet_detection = ActivityNetLocalization(
            args.gt,
            args.pd,
            tiou_thresholds=np.linspace(0.5, 0.95, 10),
            # tiou_thresholds=np.arange(0.2, 1, 0.05),  # end 1 - 0.05
            # tiou_thresholds=np.arange(0.2, 0.6, 0.1),  # 0.2 - 0.5
            verbose=True)
        mAP, average_mAP = anet_detection.evaluate()
        print(
            '[RESULTS] Performance on ActivityNet detection task.\n'
            f'mAP: {mAP}\nAverage-mAP: {average_mAP}'
        )
    # stat_centralized_file(args.pd)
    compare_gt_pd_files(args.gt, args.pd)
    print(
        '[RESULTS] Performance on ActivityNet detection task.\n'
        f'mAP: {mAP}\nAverage-mAP: {average_mAP}'
    )


if __name__ == '__main__':
    main()


# python mmaction2/report_map.py -d


# for a json file videos
# python mmaction2/report_map.py \
#     --gt /Users/eason.hung/Documents/Projects/test-something/output/ground_truth/data.json \
#     --pd /Users/eason.hung/Documents/Projects/test-something/output/predictions/merge_data.json


# for a json file video
# python mmaction2/report_map.py \
#     --gt /Users/eason.hung/Documents/Projects/test-something/external_camera/formal_output_出/Viscovery_Bread_DemoRoom_20251107_095944_annotations_ActivityNet.json \
#     --pd /Users/eason.hung/Documents/Projects/test-something/output/predictions/Viscovery_Bread_DemoRoom_20251107_095944.json

# python mmaction2/report_map.py \
#     --gt /Users/eason.hung/Documents/Projects/test-something/external_camera/formal_output_出/Viscovery_Bread_DemoRoom_20251107_100130_annotations_ActivityNet.json \
#     --pd /Users/eason.hung/Documents/Projects/test-something/output/predictions/Viscovery_Bread_DemoRoom_20251107_100130.json


# for a json file video
# python mmaction2/report_map.py \
#     --gt /Users/eason.hung/Documents/Projects/test-something/external_camera/formal_output_出/Viscovery_Bread_DemoRoom_20251107_100030_annotations_ActivityNet.json \
#     --pd /Users/eason.hung/Documents/Projects/test-something/output/predictions/Viscovery_Bread_DemoRoom_20251107_100030.json


# 1. 看太慢,
# 2. override learning rate.
# 3. Vibe parameters tuning.
# 4. bgs library;
# 5. 把 mask metric 都拉出來
# 6. 更多實驗,
# 7. Focus on 一般的情況
# 參數沒調對, 資料集太極端, 資料集不適合(x)？ => shawn 不覺得

# 現有的問題：

# 之少讓東西看起來是能用的
# apply 後的 learningRateOverride (不能用)
# background 更新太慢, foreground 炸掉. => (1 ~ 2 sec 內).


# 就這部影片的問題
# - foreground 的 mask 為什麼破碎 (為什麼-> 那我可以怎麼樣影響或調整它, 不要自己忙調 要看我歸類的問題找論文的建議)
# - SubSense foreground.

# 正常結帳行為 ->
# 畫面炸掉 -> foreground mask 炸掉
# 簡單光源的影響 -

# SuBSENSE 為什麼 background 看起來都沒有在更新
# （更新的速度真的太慢太慢 -> 但因為 ViBe 和 SuBSENSE 都很明確的在說 他的是用的是極度保守的更新 -> 為了強調不樣讓 BG 被foreground 給污染）
#  光是打在這個點上 我就覺得 這邊跟我們想要的 foreground 行為不太一樣了
#  那是否有辦法奠基在他們的設定情況之下 有辦法有效的做到不保守更新 ? 更新可以更激進

"""
Due to the conservative update strategy and neighbor-spread rules used
to update these samples, our model is resistant to shaking cameras and
intermittent object motion.
這段話說明的小鏡頭擾動 + 間歇性物體移動可以支援, 因為夠保守所以不擔心這個問題. (非常保守的估計)
"""

"""
Then, one of the neighbors of B(x) also has a 1/T probability of seeing one
of its samples replaced by this same observation. This new parameter controls
the adaptation speed of our background model: small values lead to high update
probabilities (and thus to the rapid evolution of the model) and vice-versa.
就在說明如果周圍有已經認定爲是 B(x) 的 pixel 他是有機會 1 / T 機會去影響其他人的.
"""

"""
Since new samples can only be inserted when a local pixel is recognized as background,
this approach prevents static foreground objects from being assimilated too fast.
就是因為說明了這段 只有 B(x) 的附近人可以更新, 所以 background update 速度才會慢的跟屎一樣(超級保守)

我認為這邊所認定的問題是 -> 我們其實更希望傾向是 Blind Update (背景更新盲更新)

最後在說明理論上不會被更新到 實際上放到無限長時間他還是會被靜置的物體給更新掉
In practice, noise and camouflage always cause gradual foreground erosion,
meaning that all static objects will eventually be classified as background.

要記得我是主軸 我要非常清楚每一步驟並且可以變更好的原因. 繼續想吧
"""

""" Ghost
Ghost artifacts, which are commonly defined as falsely classified background regions due
to the removal of an object from the observed scene, can be eliminated rapidly since they
share many similarities with other parts of the background.
(Spatial propagation)(spatial diffusion)
1. pros: limit cameras motion can be tolerated.

2. pros: 紋理訊息可以避免擴再到物體的邊界區域之外 (這段話需要非常仔細思考)

    note: 即使一個樣本被錯誤地從一個區域移動到另一個區域
    它在新區域中被匹配的機率也大大降低，因為 LBSP 特徵能夠檢測到邊界附近的紋理變化。
    事實上，如果前景物體的邊界清晰可見，即使它的顏色與背景相似，也可能長時間被正確分類。
"""

"""
However, using LBP-like features at the pixel
level still results in false foreground classifications in most
dynamic background regions, as textures are much harder
to match than color values.
然而，在像素層級使用類似 LBP 的特徵仍然會導致大多數動態背景區域出現錯誤的前景分類，因為紋理比顏色值更難匹配。

"""



# 現在流程
# SuBSENSE + ViBe paper 的 key point 記錄下來並且分類到對應的應該處理的情況
# 1. 光線我要解決什麼, 正常的小影子,
# 2. background 要解什麼, 我們認為 間歇性物體的更新速率太慢 想要更新快一點

# 3. foreground 要解什麼, foreground mask 炸掉的情況




# ViBe:
# 說白了還是回去看論文的理由吧
# 理由寫好寫清楚
# 從high-level 整理我要解決什麼 -> 那這段可以說明代表什麼 -> 預期可以看到的效果
# 因為要有很明確的想法
# 1.1 hard case 超強光影真的沒救
# 1.2 正常 case 為什麼 foreground 會破碎 o|r  會噴 mask 出來.


# foreground LBSP
# 1. resolution -> resize 可以降低破碎的情況, 也可以因為解析度降低能做到很快的擴散
# 1.cons: 他會受到物體與被境相似的情境影響

# 2. 超 1 -> spatial propogation
# 3. LBSP
# (炸掉 background 更新太慢.)

# 再去實作

# resolution
# 為什麼我不要直接設定一個 hold up time 0.3 (sec)
# 去決定哪一些區域是可以有機會一起從foreground 變成 background 吃掉？
# 這樣的想法是在於我們會認定商品在進入畫面之後通常會被穩定置放一段時間, 然而光影是會有變化的或是瞬間消逝
# 對於短時間路過的光影我們雖然也可拿到他的 mask. 但是因為時間長度不夠的關係 我們就不會拿該區域的pixel 去更新我的background
# 請告訴我這樣思考的所有風險：需要非常嚴謹
# 又是回到一樣的問題 我到底要怎麼樣使用？
# 高層次：Trigger -> Middle-Level : model infer object bbox -> Low-Level : Stable Unstable offer
# Block-Based 目前大部分情形可以表現得還算不錯.
# 但是 影子介入畫面的情況下
# 小物體鄰近邊界的偵測上, (小物體與背景顏色相似的情況下) 都是仍然重中之重的待解問題
# 現在影子的介入 因為我看的是區域性的變化來決定他是否有被觸發的行為.

# 如果需要更細緻的判別說某些區域是否有受到光線影響
# 邏輯到底是什麼 ?

# background subtraction




