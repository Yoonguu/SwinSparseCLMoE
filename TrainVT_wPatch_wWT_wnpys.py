#!/usr/bin/env python3
"""
주요 특징:
- **MONAI 기반 파이프라인**:
    - 패치 기반 훈련 (Patch-based Training) (예: 128x128x128)
    - 강력한 데이터 증강 (Affine, Intensity, Elastic 등)
    - 슬라이딩 윈도우 추론 (Sliding Window Inference)
    - 표준 손실 함수 (DiceCELoss)
- **WandB 로깅** 및 체크포인트 저장.

***추가***
-Whole Tumor등의 체크포인트 추가
    - compute_metrics 변경
    - wandb-project 이름 변경
    - criteria = (nn.CrossEntropyLoss().to(device), DiceLoss(include_background=True).to(device))
        -> # 배경(0) 제외하고 1,2,3 라벨에 대해서만 Dice Loss 계산하도록 설정
            criteria = (
                nn.CrossEntropyLoss().to(device),
                DiceLoss(include_background=False, softmax=True).to(device)
            )
***추가***
Brats24 + Brats 21


설치 필요:
pip install torch numpy nibabel wandb monai medpy
"""

import argparse
import glob
import os
import random
import re
import socket
import warnings
from typing import List, Tuple, Dict, Any, Union, Type

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random


try:
    import nibabel as nib
except ImportError:
    raise ImportError("Please install nibabel: pip install nibabel")

try:
    import wandb

    _wandb_available = True
except ImportError:
    _wandb_available = False

try:
    from medpy.metric.binary import hd, hd95, asd, assd
except ImportError:
    raise ImportError("Please install MedPy: pip install MedPy")

try:
    import monai
    from monai.data import Dataset, DataLoader, CacheDataset, decollate_batch, PersistentDataset
    from monai.inferers import sliding_window_inference
    from monai.losses import DiceCELoss
    from monai.losses import DiceLoss  # DiceCELoss 대신 DiceLoss 임포트
    from monai.transforms import (
        # AddChanneld,
        AsDiscrete,
        Compose,
        CropForegroundd,
        LoadImaged,
        Orientationd,
        NormalizeIntensityd,
        RandCropByPosNegLabeld,
        RandFlipd,
        RandRotate90d,
        RandShiftIntensityd,
        RandScaleIntensityd,
        RandGaussianNoised,
        RandGaussianSmoothd,
        Spacingd,
        SpatialPadd,
        CopyItemsd,
        EnsureTyped,
        EnsureType,
        EnsureChannelFirstd,
    )
    from monai.utils import set_determinism
except ImportError:
    raise ImportError("Please install MONAI: pip install monai")

def convert_brats_label(x):
    """Converts BraTS labels from 4 to 3 (for ET)."""
    return np.where(x == 4, 3, x)


class DiceLoss(nn.Module):
    def __init__(self, include_background=True, smooth=1e-5):
        super().__init__()
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits, targets):
        B, C, D, H, W = logits.shape
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=C).permute(0, 4, 1, 2, 3).float()
        if not self.include_background and C > 1:
            probs, one_hot = probs[:, 1:], one_hot[:, 1:]
        dims = (0, 2, 3, 4)
        inter = torch.sum(probs * one_hot, dims)
        denom = torch.sum(probs + one_hot, dims)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()

def info_nce_loss(emb1, emb2, temperature=0.07):
    B = emb1.size(0)
    z = torch.cat([emb1, emb2], dim=0)
    sim = torch.matmul(z, z.t())
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    labels = torch.cat([torch.arange(B, device=z.device) + B, torch.arange(B, device=z.device)])
    logits = (sim / temperature).masked_fill(mask, float('-inf'))
    return F.cross_entropy(logits, labels)

def make_second_view(x):
    x2 = x.clone()
    if torch.rand(1) < 0.8:
        scale = 1.0 + 0.1 * (2 * torch.rand(x2.size(0), 1, 1, 1, 1, device=x2.device) - 1.0)
        bias = 0.05 * (2 * torch.rand(x2.size(0), 1, 1, 1, 1, device=x2.device) - 1.0)
        x2 = x2 * scale + bias
    if torch.rand(1) < 0.5:
        x2 = torch.flip(x2, dims=[3])
    if torch.rand(1) < 0.5:
        x2 = torch.flip(x2, dims=[4])
    return x2


# def set_seed(seed: int) -> None:
#     # MONAI의 set_determinism 사용
#     set_determinism(seed)
#     warnings.warn(
#         "set_determinism_ENABLING DETERMINISM (SLOWS DOWN TRAINING)"
#     )


def set_seed(seed: int) -> None:
    # 1. 에러가 발생하는 MONAI 함수는 끕니다.
    # set_determinism(seed)

    # 2. 대신 기본 라이브러리로 시드를 직접 고정합니다. (이게 중요!)


    random.seed(seed)  # <--- 이것 덕분에 데이터 Split 순서가 유지됨
    np.random.seed(seed)  # <--- Numpy 연산 재현성 유지
    torch.manual_seed(seed)  # <--- PyTorch 연산 재현성 유지
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    warnings.warn("Manual seeding applied to avoid MONAI OverflowError.")


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Dict[str, Any]:
    """
    BraTS Regions (WT, TC, ET) + Detailed Class-wise Metrics (Jaccard, HD, ASD, etc.)
    """
    # 1. 결과 저장소 초기화
    metrics = {
        # --- [A] BraTS Standard Regions (리더보드/논문 비교용) ---
        "dice_WT": [], "dice_TC": [], "dice_ET": [],
        "hd_WT": [], "hd_TC": [], "hd_ET": [],
        "hd95_WT": [], "hd95_TC": [], "hd95_ET": [],
        "asd_WT": [], "asd_TC": [], "asd_ET": [],
        "assd_WT": [], "assd_TC": [], "assd_ET": [],

        # --- [B] Class-wise Detailed Metrics (요청하신 상세 지표) ---
        "dice_class": np.zeros(num_classes),
        "jaccard_class": np.zeros(num_classes),
        "precision_class": np.zeros(num_classes),
        "sensitivity_class": np.zeros(num_classes),
        "specificity_class": np.zeros(num_classes),
        "hd_class": np.zeros(num_classes),
        "hd95_class": np.zeros(num_classes),
        "asd_class": np.zeros(num_classes),
        "assd_class": np.zeros(num_classes)
    }

    # (B, C, D, H, W) -> (B, D, H, W)
    preds_argmax = torch.argmax(preds, dim=1).cpu().numpy()
    targets_np = targets.squeeze(1).cpu().numpy()

    batch_size = preds_argmax.shape[0]

    for b in range(batch_size):
        p = preds_argmax[b]
        t = targets_np[b]

        # ==============================================================================
        # Part 1: BraTS Region-wise Calculation (WT, TC, ET)
        # ==============================================================================
        # WT: Whole Tumor (1+2+3), TC: Tumor Core (1+3), ET: Enhancing Tumor (3)
        regions = {
            "WT": (np.isin(p, [1, 2, 3]), np.isin(t, [1, 2, 3])),
            "TC": (np.isin(p, [1, 3]), np.isin(t, [1, 3])),
            "ET": (p == 3, t == 3)
        }

        for region_name, (pred_mask, tgt_mask) in regions.items():
            # Dice
            intersection = (pred_mask & tgt_mask).sum()
            union = pred_mask.sum() + tgt_mask.sum()
            dice_val = (2.0 * intersection) / (union + 1e-8) if union > 0 else 1.0
            metrics[f"dice_{region_name}"].append(dice_val)

            # HD95 (데이터가 존재하는 경우에만 계산)
            if pred_mask.sum() > 0 and tgt_mask.sum() > 0:
                try:
                    metrics[f"hd_{region_name}"].append(hd(pred_mask, tgt_mask))
                    metrics[f"hd95_{region_name}"].append(hd95(pred_mask, tgt_mask))
                    metrics[f"asd_{region_name}"].append(asd(pred_mask, tgt_mask))
                    metrics[f"assd_{region_name}"].append(assd(pred_mask, tgt_mask))
                except:
                    metrics[f"hd_{region_name}"].append(np.nan)
                    metrics[f"hd95_{region_name}"].append(np.nan)
                    metrics[f"asd_{region_name}"].append(np.nan)
                    metrics[f"assd_{region_name}"].append(np.nan)
            else:
                metrics[f"hd_{region_name}"].append(np.nan)
                metrics[f"hd95_{region_name}"].append(np.nan)
                metrics[f"asd_{region_name}"].append(np.nan)
                metrics[f"assd_{region_name}"].append(np.nan)


        # ==============================================================================
        # Part 2: Class-wise Calculation (0, 1, 2, 3) - 요청하신 모든 지표 포함
        # ==============================================================================
        for c in range(num_classes):
            p_cls, t_cls = (p == c), (t == c)

            # Basic Sets
            intersection = (p_cls & t_cls).sum()
            union_set = (p_cls | t_cls).sum()
            pred_sum = p_cls.sum()
            tgt_sum = t_cls.sum()

            # 1. Overlap Metrics
            # Dice
            metrics["dice_class"][c] += (2.0 * intersection) / (pred_sum + tgt_sum + 1e-8)
            # Jaccard (IoU)
            metrics["jaccard_class"][c] += intersection / (union_set + 1e-8)

            # 2. Confusion Matrix Metrics
            tp = intersection
            fp = pred_sum - tp
            fn = tgt_sum - tp
            tn = p.size - (tp + fp + fn)

            # Sensitivity (Recall)
            metrics["sensitivity_class"][c] += tp / (tp + fn + 1e-8)
            # Specificity
            metrics["specificity_class"][c] += tn / (tn + fp + 1e-8)
            # Precision
            metrics["precision_class"][c] += tp / (tp + fp + 1e-8)
            # Sensitivity
            metrics["sensitivity_class"][c] += tp / (tp + fn + 1e-8)

            # 3. Surface Distance Metrics (MedPy) - 시간이 오래 걸릴 수 있음
            if pred_sum > 0 and tgt_sum > 0:
                try:
                    metrics["hd_class"][c] += hd(p_cls, t_cls)
                    metrics["hd95_class"][c] += hd95(p_cls, t_cls)
                    metrics["asd_class"][c] += asd(p_cls, t_cls)
                    metrics["assd_class"][c] += assd(p_cls, t_cls)
                except:
                    # 계산 실패 시(예: 너무 작은 영역) NaN 누적 대신 0 혹은 무시
                    pass
            else:
                # 둘 중 하나라도 없으면 거리는 정의되지 않음 (0 처리 혹은 NaN)
                pass

    # ==============================================================================
    # Final Aggregation (Average over Batch)
    # ==============================================================================
    final_metrics = {}
    region_metric_keys = ["dice", "hd", "hd95", "asd", "assd"]
    for m_name in region_metric_keys:
        for r_name in ["WT", "TC", "ET"]:
            key = f"{m_name}_{r_name}"
            final_metrics[key] = np.nanmean(metrics[key])

    # 2. Class Mean (누적합 / 배치사이즈)
    class_keys = [k for k in metrics.keys() if "_class" in k]
    for key in class_keys:
        final_metrics[key] = metrics[key] / batch_size

    # 3. Summary Metric (Leaderboard Standard)
    final_metrics["dice_mean"] = (final_metrics["dice_WT"] + final_metrics["dice_TC"] + final_metrics["dice_ET"]) / 3.0

    return final_metrics



def find_file_lists_as_dicts(data_dir: str) -> List[Dict[str, str]]:
    """
    데이터 디렉토리를 스캔하여 MONAI가 요구하는 딕셔너리 리스트를 생성합니다.
    지원 포맷:
      - BraTS 2024: BraTS-GLI-XXXXX-XXX (t1n, t1c, t2w, t2f, seg)
      - BraTS 2021: BraTS2021_XXXXX (t1, t1ce, t2, flair, seg)
    """
    all_files = glob.glob(os.path.join(data_dir, "**", "*.nii*"), recursive=True)

    file_dict_by_case = {}

    for fpath in all_files:
        if "validation_data" in fpath.lower(): continue

        fname = os.path.basename(fpath).lower()

        # 1. Case ID Extraction
        # BraTS 2024 Pattern: BraTS-GLI-00000-000
        match_24 = re.search(r"(BraTS-GLI-\d{5}-\d{3})", fname, re.IGNORECASE)
        # BraTS 2021 Pattern: BraTS2021_00000
        match_21 = re.search(r"(BraTS2021_\d{5})", fname, re.IGNORECASE)

        if match_24:
            case_id = match_24.group(1)
        elif match_21:
            case_id = match_21.group(1)
        else:
            continue

        if case_id not in file_dict_by_case:
            file_dict_by_case[case_id] = {}

        # 2. Modality Matching
        # 순서 중요: t1c/t1ce를 t1보다 먼저 검사해야 함 (t1ce가 t1을 포함하므로)

        # T1 Contrast (t1c, t1ce)
        if "t1c" in fname or "t1ce" in fname:
            file_dict_by_case[case_id]["t1c"] = fpath

        # T1 Native (t1n, t1)
        # 2021은 _t1.nii.gz, 2024는 -t1n.nii.gz
        elif "t1n" in fname or "t1" in fname:
            file_dict_by_case[case_id]["t1n"] = fpath

        # T2 FLAIR (t2f, flair)
        elif "t2f" in fname or "flair" in fname:
            file_dict_by_case[case_id]["t2f"] = fpath

        # T2 Weighted (t2w, t2)
        # 2021은 _t2.nii.gz, 2024는 -t2w.nii.gz
        elif "t2w" in fname or "t2" in fname:
            file_dict_by_case[case_id]["t2w"] = fpath

        # Segmentation (seg)
        elif "seg" in fname:
            file_dict_by_case[case_id]["seg"] = fpath

    # t1n이 없는 경우 t1c로 대체 (4채널 맞추기용, 2021 데이터는 t1이 있으므로 이 로직은 주로 예외상황용)
    final_list = []
    modalities = ["t1c", "t1n", "t2f", "t2w", "seg"]
    for case_id, files in file_dict_by_case.items():
        if "t1n" not in files and "t1c" in files:
            files["t1n"] = files["t1c"]
            warnings.warn(f"Case {case_id}: Missing 't1n', using 't1c' as placeholder.")

        if all(m in files for m in modalities):
            final_list.append(files)
        else:
            # 디버깅: 빠진 모달리티 확인 가능
            # missing = [m for m in modalities if m not in files]
            # print(f"Skipping case {case_id}, missing: {missing}")
            pass

    print(f"Found {len(final_list)} complete cases.")
    return final_list


def get_train_transform(args: argparse.Namespace) -> Compose:
    """
    SOTA 훈련을 위한 MONAI 변환 파이프라인
    """
    roi_size = tuple(args.roi_size)

    return Compose(
        [
            LoadImaged(keys=["t1c", "t1n", "t2f", "t2w", "seg"]),
            EnsureChannelFirstd(keys=["t1c", "t1n", "t2f", "t2w", "seg"]),
            # 4개의 모달리티를 4채널 텐서 하나로 결합
            monai.transforms.ConcatItemsd(keys=["t1c", "t1n", "t2f", "t2w"], name="image"),
            # (1, D, H, W) -> (4, D, H, W)
            # monai.transforms.DeleteItemsd(keys=["t1c", "t1n", "t2f", "t2w"]),
            monai.transforms.DeleteItemsd(
                keys=["t1c", "t1n", "t2f", "t2w", "t1c_meta_dict", "t1n_meta_dict", "t2f_meta_dict", "t2w_meta_dict"]),
            # 데이터 전처리
            Orientationd(keys=["image", "seg"], axcodes="RAS"),
            Spacingd(keys=["image", "seg"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),

            # 라벨 값 변환 (BRATS: 0, 1, 2, 4 -> 0, 1, 2, 3)
            monai.transforms.CopyItemsd(keys=["seg"], times=1, names=["seg_orig"]),  # 원본 백업
            # monai.transforms.Lambdad(keys=["seg"], func=lambda x: np.where(x == 4, 3, x)),
            monai.transforms.Lambdad(keys=["seg"], func=convert_brats_label),  # 'lambda' 대신 'convert_brats_label' 함수 사용

            # 전경(Foreground) 기준으로 크롭
            CropForegroundd(keys=["image", "seg"], source_key="image"),

            # 정규화
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

            SpatialPadd(keys=["image", "seg"], spatial_size=roi_size, mode="constant", constant_values=0),


            # 패치 추출 (SOTA의 핵심)
            RandCropByPosNegLabeld(
                keys=["image", "seg"],
                label_key="seg",
                spatial_size=roi_size,
                pos=1.0,  # 100% 전경 샘플링 (pos=0.9, neg=0.1도 좋음)
                neg=0.0,
                num_samples=1,  # DataLoader가 배치 크기만큼 샘플링
                image_key="image",
                image_threshold=0,
            ),

            # 데이터 증강
            RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=0),  # D
            RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=1),  # H
            RandFlipd(keys=["image", "seg"], prob=0.5, spatial_axis=2),  # W
            RandRotate90d(keys=["image", "seg"], prob=0.5, max_k=3),

            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            RandGaussianNoised(keys="image", prob=0.15, mean=0.0, std=0.1),
            RandGaussianSmoothd(keys="image", prob=0.15, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),

            EnsureTyped(keys=["image", "seg"], dtype=torch.float32),

        ]
    )


def get_val_transform(args: argparse.Namespace) -> Compose:
    """
    SOTA 검증/테스트를 위한 MONAI 변환 파이프라인
    (증강 및 패치 크롭 없음)
    """
    return Compose(
        [
            CopyItemsd(keys=["seg"], times=1, names=["original_path"]),
            LoadImaged(keys=["t1c", "t1n", "t2f", "t2w", "seg"]),
            EnsureChannelFirstd(keys=["t1c", "t1n", "t2f", "t2w", "seg"]),
            monai.transforms.ConcatItemsd(keys=["t1c", "t1n", "t2f", "t2w"], name="image"),
            monai.transforms.DeleteItemsd(keys=["t1c", "t1n", "t2f", "t2w"]),

            Orientationd(keys=["image", "seg"], axcodes="RAS"),
            Spacingd(keys=["image", "seg"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),

            # monai.transforms.Lambdad(keys=["seg"], func=lambda x: np.where(x == 4, 3, x)),
            monai.transforms.Lambdad(keys=["seg"], func=convert_brats_label),  # 'lambda' 대신 'convert_brats_label' 함수 사용

            CropForegroundd(keys=["image", "seg"], source_key="image"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),

            EnsureTyped(keys=["image", "seg"], dtype=torch.float32),

        ]
    )


def get_default_data_dir(dataset_version='24'):
    hostname = socket.gethostname().lower()

    # BraTS 2024
    if dataset_version == '24':
        if 'bmis' in hostname: return '/home/bmis1/DATA2/YG/Dataset/Brain/BRATS2024/BraTS-GLI'
        if 'gm-a6000' in hostname: return '/home/gmadmin/DATA2/Song/Dataset/Brain/BRATS/BraTS2024/BraTS-GLI'
        if 'supermoon' in hostname: return '/home/leebr/Desktop/YG/Dataset/Brain/BRATS2024/BraTS-GLI'
        if 'mega' in hostname: return '/home/mega/Desktop/Data1/Dataset/Brain/BRATS2024/BraTS-GLI'
        return 'E:\\Brain\\BRATS2024\\BraTS-GLI'

    # BraTS 2021
    elif dataset_version == '21':
        if 'bmis' in hostname: return '/home/bmis1/DATA2/YG/Dataset/Brain/BraTS2021/TrainingData'
        if 'gm-a6000' in hostname: return '/home/gmadmin/DATA2/Song/Dataset/Brain/BRATS/BraTS2021/TrainingData'
        if 'supermoon' in hostname: return '/home/leebr/Desktop/YG/Dataset/Brain/BraTS2021/TrainingData'
        if 'mega' in hostname: return '/home/mega/Desktop/Data1/Dataset/Brain/BraTS2021/TrainingData'

        # 사용자가 지정한 로컬 경로
        return 'E:\\Brain\\BraTS2021\\TrainingData'

    # 기본값
    return 'E:\\Brain\\BRATS2024\\BraTS-GLI'




def create_next_result_folder(base_dir, hostname, args, str_debug):
    os.makedirs(base_dir, exist_ok=True)
    # [수정] 파일명 패턴에 BraTS 버전 추가 (Brats{args.brats_version}_)
    pattern = re.compile(rf"^{str_debug}Brats{args.brats_version}_SwinSparseCLMoE_Waux_{re.escape(hostname)}_a{args.alpha}_b{args.beta}_c{args.lambda_c}_ch{args.base_channel}_bat{args.batch_size}_EC{args.expert_count}_TK{args.top_k}_drop{args.droprate}_roi{args.roi_size[0]}_(\d+)$")
    existing = [int(m.group(1)) for name in os.listdir(base_dir) if (m := pattern.match(name))]
    next_num = max(existing) + 1 if existing else 1
    file_name = f"{str_debug}Brats{args.brats_version}_SwinSparseCLMoE_Waux_{hostname}_a{args.alpha}_b{args.beta}_c{args.lambda_c}_ch{args.base_channel}_bat{args.batch_size}_EC{args.expert_count}_TK{args.top_k}_drop{args.droprate}_roi{args.roi_size[0]}_{next_num}"
    new_folder = os.path.join(base_dir, file_name)
    os.makedirs(new_folder)
    print(f"Created result folder: {new_folder}")
    return new_folder, file_name


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_dice):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_dice": best_dice
    }
    if scheduler:
        ckpt["scheduler"] = scheduler.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt: scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_dice = ckpt.get("best_dice", 0.0)
    return start_epoch, best_dice



def train_one_epoch(model, loader, optimizer, criteria, device, alpha, beta, lambda_c, max_steps):
    model.train()
    # running_loss = 0.0
    running_loss, running_dice, running_ce, running_infonce = 0.0, 0.0, 0.0, 0.0
    step = 0
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    # scaler = torch.amp.GradScaler(device='cuda', enabled=torch.cuda.is_available())

    # criteria 분리
    ce_criterion, dice_criterion = criteria

    for batch_data in loader:
        # print(f'step : {step}')
        # print(f'\t num_data : {num_data}')
        # if num_data > 2:
        #     break

        images, masks = (
            batch_data["image"].to(device, dtype=torch.float32),
            batch_data["seg"].to(device, dtype=torch.long),
        )
        # print(f'images.shape : {images.shape}')
        # print(f'masks.shape : {masks.shape}')
        optimizer.zero_grad(set_to_none=True)

        # with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
            x2 = make_second_view(images)
            # _, logits, proj1, proj2 = model(images, x2)
            seg_feat, logits, proj1, proj2, aux_loss = model(images, x2)
            # print(f'logits.shape : {logits.shape}')
            # loss = loss_fn(logits, masks)
            # MONAI의 DiceLoss는 (B, C, D, H, W) 형태의 logits와
            # (B, 1, D, H, W) 형태의 masks를 받습니다 (to_onehot_y=True 설정 시).
            # dice_loss_val = dice_criterion(logits, masks)
            # ce_loss_val = ce_criterion(logits, masks)
            # loss = alpha * dice_loss_val + beta * ce_loss_val
            masks_squeezed = masks.squeeze(1)  # [B, 1, D, H, W] -> [B, D, H, W]
            dice_loss_val = dice_criterion(logits, masks_squeezed)
            ce_loss_val = ce_criterion(logits, masks_squeezed)
            infonce_loss_val = info_nce_loss(proj1, proj2)
            #aux_loss는 이미 aux_loss_weight(기본값 0.01)가 곱해진 상태로 moe_head에서 반환됩니다. 만약 학습이 불안정하다면 이 가중치를 조절하거나, aux_loss에 추가적인 계수를 곱해 조절해야함
            loss = alpha * dice_loss_val + beta * ce_loss_val + lambda_c * infonce_loss_val+aux_loss


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_dice += dice_loss_val.item() * batch_size
        running_ce += ce_loss_val.item() * batch_size
        running_infonce += infonce_loss_val.item() * batch_size

        step += 1

        if max_steps and step >= max_steps:
            break

    num_samples = step * loader.batch_size if step > 0 else 0
    if num_samples == 0:
        print("WARNING: No data processed in train_one_epoch.")
        # return 0.0
        return 0.0, 0.0, 0.0, 0.0

    # return running_loss / num_samples
    return (
        running_loss / num_samples,
        running_dice / num_samples,
        running_ce / num_samples,
        running_infonce / num_samples,
    )


@torch.no_grad()
def evaluate(model, loader, device, num_classes, roi_size, sw_batch_size=4, is_cl=True,
             save_preds=False, output_dir=None, max_preds_to_save=4, is_test=False):
    model.eval()
    inferer = sliding_window_inference

    # 모든 상세 메트릭 리스트 통합 초기화
    results_storage = {
        # [A] BraTS Region-wise (v14 핵심 지표)
        "dice_WT": [], "dice_TC": [], "dice_ET": [],
        "hd95_WT": [], "hd95_TC": [], "hd95_ET": [],

        # [B] Average Metrics (배경 제외 클래스 1,2,3 평균 - v10 상세 지표 포함)
        # "avg_dc": [], "avg_hd": [], "avg_hd95": [], "avg_jc": [],
        # "avg_asd": [], "avg_assd": [], "avg_prc": [], "avg_rcl": [],
        # "avg_sensi": [], "avg_speci": [],

        # [C] Full Lists (샘플별 전체 데이터)
        "dice_class_list": [],  # 클래스 0,1,2,3 전체 Dice
        "hd_class_list": [],
        "hd95_class_list": [],
        "jc_class_list": [],
        "asd_class_list": [],
        "assd_class_list": [],
        "prc_class_list": [],  # Precision
        "rcl_class_list": [],  # Recall (Sensitivity)
        "speci_class_list": [],  # Specificity




        "dice_mean_list": [],  # (WT+TC+ET)/3 평균
        "hd_mean_list": [],
        "hd95_mean_list": [],
        "asd_mean_list": [],
        "assd_mean_list": []
    }

    for batch_data in loader:
        images, masks = (
            batch_data["image"].to(device, dtype=torch.float32),
            batch_data["seg"].to(device, dtype=torch.long),
        )

        # 모델 타입(CL 여부)에 따른 predictor 설정
        if is_cl:
            predictor = lambda x: model(x, x)[1]
        else:
            predictor = lambda x: model(x)[0]

        # 슬라이딩 윈도우 추론 실행
        logits = inferer(
            inputs=images,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=predictor,
            overlap=0.5,
            mode="gaussian"
        )

        # 로짓을 stack하여 compute_metrics로 전달
        preds_decollated = decollate_batch(logits)
        preds_tensor = torch.stack(preds_decollated).to(device)

        # 메트릭 계산 실행 (v14의 compute_metrics 활용)
        # 해당 함수는 이미 dice_class, jaccard_class 등을 반환함
        m = compute_metrics(preds_tensor, masks, num_classes)

        # --- 데이터 누적 ---
        # 1. 지역별 지표 저장
        results_storage["dice_WT"].append(m["dice_WT"])
        results_storage["dice_TC"].append(m["dice_TC"])
        results_storage["dice_ET"].append(m["dice_ET"])
        results_storage["hd95_WT"].append(m["hd95_WT"])
        results_storage["hd95_TC"].append(m["hd95_TC"])
        results_storage["hd95_ET"].append(m["hd95_ET"])

        # 2. 클래스별 전체 리스트 및 BraTS 평균 저장
        results_storage["dice_class_list"].append(m["dice_class"])
        results_storage["dice_class_list"].append(m["dice_class"])
        results_storage["hd_class_list"].append(m["hd_class"])
        results_storage["hd95_class_list"].append(m["hd95_class"])
        results_storage["jc_class_list"].append(m["jaccard_class"])
        results_storage["asd_class_list"].append(m["asd_class"])
        results_storage["assd_class_list"].append(m["assd_class"])
        results_storage["prc_class_list"].append(m["precision_class"])
        results_storage["rcl_class_list"].append(m["sensitivity_class"])
        results_storage["speci_class_list"].append(m["specificity_class"])



        # results_storage["dice_mean_list"].append(m["dice_mean"])
        results_storage["dice_mean_list"].append([m["dice_WT"], m["dice_TC"], m["dice_ET"]])
        results_storage["hd_mean_list"].append([m["hd_WT"], m["hd_TC"], m["hd_ET"]])
        results_storage["hd95_mean_list"].append([m["hd95_WT"], m["hd95_TC"], m["hd95_ET"]])
        results_storage["asd_mean_list"].append([m["asd_WT"], m["asd_TC"], m["asd_ET"]])
        results_storage["assd_mean_list"].append([m["assd_WT"], m["assd_TC"], m["assd_ET"]])


        '''
        # 3. 배경 제외 상세 평균 지표 계산 (라벨 1, 2, 3)
        valid_idx = slice(1, None)
        metrics_mapping = [
            ('avg_dc', 'dice_class'), ('avg_hd', 'hd_class'),
            ('avg_hd95', 'hd95_class'), ('avg_jc', 'jaccard_class'),
            ('avg_asd', 'asd_class'), ('avg_assd', 'assd_class'),
            ('avg_sensi', 'sensitivity_class'), ('avg_speci', 'specificity_class')
        ]

        # v14 compute_metrics 결과에 없는 prc, rcl은 sensi 등과 매칭하거나 필요시 추가 계산
        for store_key, m_key in metrics_mapping:
            # print(f'store_key : {store_key}') #avg_dc
            # print(f'm_key : {m_key}') #dice_class

            if m_key in m:
                avg_val = np.nanmean(m[m_key][valid_idx])
                results_storage[store_key].append(avg_val)
        '''

    # --- [수정] .npy 파일 저장 및 WandB 업로드 로직 ---
    if is_test and output_dir:
        npy_path = os.path.join(output_dir, 'npy_results')
        os.makedirs(npy_path, exist_ok=True)

        # 1. 파일 저장
        saved_files = []
        for k, v in results_storage.items():
            if len(v) > 0:
                file_full_path = os.path.join(npy_path, f'{k}.npy')
                np.save(file_full_path, np.array(v))
                saved_files.append(file_full_path)

        # 2. WandB Artifact 업로드 (WandB가 활성화된 경우)
        if wandb.run is not None:
            artifact = wandb.Artifact(
                name=f"test_metrics_{wandb.run.id}",
                type="evaluation_results",
                description="Detailed per-case metrics in .npy format"
            )
            # npy_results 폴더 전체를 artifact에 추가
            artifact.add_dir(npy_path)
            wandb.log_artifact(artifact)
            print(f"Successfully uploaded {len(saved_files)} .npy files to WandB Artifacts.")

    # 화면 출력용 최종 평균값 반환
    aggregated = {k: np.nanmean(v, axis=0) for k, v in results_storage.items() if len(v) > 0}

    # WandB 호환성을 위해 dice_mean 키 유지
    aggregated["dice_mean"] = np.nanmean(results_storage["dice_mean_list"])
    return aggregated



# =================================================================================
# Main Execution Logic
# =================================================================================

def main(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser(description="Train Patch based SOTA pipeline.")
    parser.add_argument("--brats-version", type=str, default="21", choices=["21", "24"],
                        help="BraTS dataset version (21 or 24)")
    parser.add_argument("--data-dir", type=str, default=None, help="BraTS data directory.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for patch-based training.")
    parser.add_argument("--dataset-type", type=str, default="simple",
                        choices=["cache", "persistent", "simple"],
                        help="Choose dataset type: 'cache' (RAM), 'persistent' (Disk Cache), 'simple' (No Cache)")

    # [추가] PersistentDataset 사용 시 캐시 저장 경로 (기본값은 결과 폴더 내 생성)
    parser.add_argument("--persistent-cache-dir", type=str, default=None,
                        help="Directory for PersistentDataset cache. If None, creates inside result dir.")
    # gmadmin 8돌아가는데 16은 안돌아감
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--class-names", type=str, default="Background,NCR/NET,ED,ET")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-worker", type=int, default=0)#4

    parser.add_argument("--alpha", type=float, default=1.0, help="Dice loss weight.")
    parser.add_argument("--beta", type=float, default=1.0, help="CE loss weight (DiceCELoss is 1:1).")
    parser.add_argument("--lambda-c", type=float, default=0.1, help="InfoNCE loss weight.")
    parser.add_argument("--base-channel", type=int, default=24,
                        help="Base channel for model, default for Mamba_Moe_CL is 32")
    parser.add_argument("--droprate", type=float, default=0.2,
                        help="droprate")

    # SOTA 파이프라인 파라미터
    parser.add_argument("--roi-size", type=int, nargs=3, default=[128, 128, 128],
                        help="ROI size for patch-based training (D, H, W)")
    parser.add_argument("--cache-rate", type=float, default=1.0, help="MONAI CacheDataset rate (1.0 = all in RAM)")

    # 기존 파라미터
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path to resume from.")
    parser.add_argument("--scheduler", type=str, choices=["cosine", "plateau", "none"], default="plateau")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-threshold", type=float, default=1e-4)

    parser.add_argument("--debug", action="store_true", help="Shrink data & steps, disable wandb.")
    parser.add_argument("--debug-samples", type=int, default=16, help="How many samples when --debug.")
    parser.add_argument("--debug-steps", type=int, default=8, help="How many train steps per epoch when --debug.")
    parser.add_argument("--debug-epochs", type=int, default=8, help="Epochs when debug is applied.")

    parser.add_argument("--gpu-num", type=int, default=0)
    parser.add_argument("--wandb", type=lambda x: str(x).lower() in {'1', 'true', 'yes'}, default=True)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-id", type=str, default=None, help="WandB Run ID to resume.")
    parser.add_argument("--early-stop-patience", type=int, default=30,
                        help="Epochs to wait for val_dice improvement before stopping. 0 to disable.")

    parser.add_argument("--save-predictions",
                        dest="save_predictions",
                        action="store_true",
                        help="Save predicted masks for the final test set (Default: False).")

    parser.add_argument("--expert-count", type=int, default=4, help="Number of experts in MoE layer.")
    parser.add_argument("--top-k", type=int, default=2, help="Number of experts to route to for each token.")
    parser.add_argument("--aux-loss-weight", type=float, default=1e-2, help="Weight for MoE auxiliary loss.")


    args = parser.parse_args(argv)

    set_seed(args.seed)

    # 1. 호스트네임 먼저 파악
    hostname = socket.gethostname().lower()
    short_host = hostname.split('-')[0] if '-' in hostname else hostname


    if args.data_dir is None:
        args.data_dir = get_default_data_dir(args.brats_version)

    # 2. WandB 프로젝트 명칭 설정
    if args.wandb_project is None:
        args.wandb_project = f"{short_host}_Brats{args.brats_version}_Patch_wWT"

    # 3. 디버그 모드 처리
    str_debug = 'debug_' if args.debug else ''
    if args.debug:
        args.epochs = args.debug_epochs
        args.wandb_project = f"debug_{args.wandb_project}"
    # 4. 결과 폴더 및 결과 파일명 생성
    result_base_name = f"Results_{args.wandb_project}"


    # Resume 모드 처리 및 폴더 생성 로직
    if args.resume and os.path.exists(args.resume):
        # args.resume 경로 예시: ".../Run_Name_1/checkpoint/best_dice.pt"

        # 1. 체크포인트 파일이 있는 폴더 (.../checkpoint)
        ckpt_dir = os.path.dirname(args.resume)

        # 2. 그 상위 폴더가 결과 폴더 (.../Run_Name_1)
        dir_results = os.path.dirname(ckpt_dir)

        # 3. 폴더 이름을 Run 이름으로 사용
        file_name = os.path.basename(dir_results)

        print(f"♻️ Resuming training in existing folder: {dir_results}")
    else:
        # Resume이 아니면 새 폴더 생성
        dir_results, file_name = create_next_result_folder(result_base_name, hostname, args, str_debug)
    # --- [수정 끝] ---

    print(f'dir_results : {dir_results}')

    dir_checkpoint = os.path.join(dir_results, 'checkpoint')
    os.makedirs(dir_checkpoint, exist_ok=True)

    print("Loading and splitting data...")
    data_dicts = find_file_lists_as_dicts(args.data_dir)
    print(f'len(data_dicts) : {len(data_dicts)}')

    random.shuffle(data_dicts)
    if args.debug:
        data_dicts = data_dicts[:args.debug_samples]


    train_end = int(0.8 * len(data_dicts))
    val_end = train_end + int(0.1 * len(data_dicts))

    train_files = data_dicts[:train_end]
    val_files = data_dicts[train_end:val_end]
    test_files = data_dicts[val_end:]

    print(f"Dataset split: Train={len(train_files)}, Validation={len(val_files)}, Test={len(test_files)}")

    # MONAI 변환기 생성
    train_transform = get_train_transform(args)
    val_transform = get_val_transform(args)

    if args.dataset_type == "cache":
        print(f"Loading training data using [CacheDataset] (RAM usage: {args.cache_rate * 100}%)...")
        train_ds = CacheDataset(
            data=train_files,
            transform=train_transform,
            cache_rate=args.cache_rate,
            num_workers=args.num_worker
        )
        print("CacheDataset loaded.")

    elif args.dataset_type == "persistent":
        # 캐시 경로 설정: 사용자가 지정하지 않았으면 결과 폴더(dir_results) 안에 'persistent_cache' 생성
        if args.persistent_cache_dir:
            cache_dir = args.persistent_cache_dir
        else:
            cache_dir = os.path.join(dir_results, "persistent_cache")

        os.makedirs(cache_dir, exist_ok=True)
        print(f"Loading training data using [PersistentDataset] (Disk Cache at: {cache_dir})...")

        train_ds = PersistentDataset(
            data=train_files,
            transform=train_transform,
            cache_dir=cache_dir
        )
        print("PersistentDataset loaded.")

    else:  # args.dataset_type == "simple"
        print("Loading training data using [Dataset] (No Caching, Standard Disk I/O)...")
        train_ds = Dataset(
            data=train_files,
            transform=train_transform
        )
        print("Standard Dataset loaded.")

    print(f"len(train_ds) : {len(train_ds)}")

    val_ds = Dataset(data=val_files, transform=val_transform)
    test_ds = Dataset(data=test_files, transform=val_transform)
    print(f"len(val_ds) : {len(val_ds)}")
    print(f"len(test_ds) : {len(test_ds)}")

    # 데이터 로더 생성
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_worker
    )

    # 검증/테스트는 배치 크기 1 (슬라이딩 윈도우 추론을 위해)
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_worker
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_worker
    )

    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else "cpu")

    try:
        import SwinSparseCLMoE as model_lib
    except ImportError as e:
        raise ImportError("Could not import SwinSparseCLMoE.py") from e

    model = model_lib.SwinSparseCLMOE(
        in_channels=4,
        out_channels=args.num_classes,
        base_dim=args.base_channel,
        expert_count=args.expert_count,
        top_k=args.top_k,
        aux_loss_weight=args.aux_loss_weight,
        drop_path_rate=args.droprate
    ).to(device)

    # 손실 함수 (Dice + CrossEntropy)
    # loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)

    # 손실 함수를 2개로 분리
    # criteria = (monai.losses.DiceLoss(to_onehot_y=True, softmax=True, include_background=True),nn.CrossEntropyLoss().to(device))
    # criteria = (nn.CrossEntropyLoss().to(device), DiceLoss(include_background=True).to(device))
    criteria = (
        nn.CrossEntropyLoss().to(device),
        DiceLoss(include_background=False).to(device)
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=args.plateau_factor,
                                                               patience=args.plateau_patience, min_lr=args.min_lr,
                                                               threshold=args.plateau_threshold)

    use_wandb = args.wandb
    if use_wandb:
        if not _wandb_available: raise ImportError("Install wandb or run with --wandb False")
        # wandb.init(project=args.wandb_project, name=file_name, config=vars(args))
        wandb.init(
            project=args.wandb_project,
            name=file_name if args.wandb_id is None else None,  # ID 지정 시 이름 자동 매칭 권장
            config=vars(args),
            id=args.wandb_id,  # 기존 Run ID 입력
            resume="allow"  # ID가 있으면 잇고, 없으면 새로 생성
        )
        wandb.watch(model, log="all", log_freq=100)

    best_dice = 0.0
    start_epoch = 0
    early_stop_counter = 0  # Initialize early stopping counter

    if args.resume:
        start_epoch, best_dice = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        print(f"Resuming from epoch {start_epoch} with best dice {best_dice:.4f}")

    for epoch in range(start_epoch, args.epochs):
        max_steps = args.debug_steps if args.debug else None

        # avg_loss = train_one_epoch(
        #     model, train_loader, optimizer, loss_fn, device, max_steps
        # )
        # 3개 값 반환 및 alpha, beta 전달
        avg_loss, avg_dice, avg_ce, avg_infonce = train_one_epoch(
            model, train_loader, optimizer, criteria, device, args.alpha, args.beta,  args.lambda_c,max_steps
        )
        print(f'train_one_epoch is finished')

        # Validation
        val_metrics = evaluate(
            model, val_loader, device, args.num_classes,
            roi_size=tuple(args.roi_size),
            is_cl=True,
            sw_batch_size=args.batch_size,
            save_preds=False
        )
        val_loss = (1 - val_metrics['dice_mean'])
        mean_dice = val_metrics.get("dice_mean", 0.0)

        # if scheduler:
        #     scheduler.step()
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()


        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {avg_loss:.4f} | Val Dice: {mean_dice:.4f} | LR: {current_lr:.6f}"
        )

        if use_wandb:
            log_data = {
                "train/total_loss": avg_loss,
                "train/dice_loss": avg_dice,
                "train/ce_loss": avg_ce,
                "train/infonce_loss": avg_infonce,
                "train/lr": current_lr,
                "epoch": epoch
            }
            for name, value in val_metrics.items():
                if isinstance(value, np.ndarray) and "all" not in name:
                    for i, v in enumerate(value): log_data[f"eval/{name}_cls{i}"] = v
                elif not isinstance(value, np.ndarray):
                    log_data[f"eval/{name}"] = value
            wandb.log(log_data)

        path_best_dice = os.path.join(dir_checkpoint, "best_by_mean_dice.pt")
        if mean_dice > best_dice:
            best_dice = mean_dice
            save_checkpoint(path_best_dice, model, optimizer, scheduler, epoch, best_dice)
            print(f"  [CKPT] New best mean Dice: {mean_dice:.4f} -> saved.")
            early_stop_counter = 0  # Reset counter on improvement
        else:
            early_stop_counter += 1
            print(f"  [Early Stop] No improvement for {early_stop_counter} / {args.early_stop_patience} epochs.")

            # --- Early Stopping Check ---
            if args.early_stop_patience > 0 and early_stop_counter >= args.early_stop_patience:
                print(f"*** EARLY STOPPING triggered after {epoch + 1} epochs due to no improvement. ***")
                break

    print("\nTraining finished. Starting final evaluation on the Test Set...")
    best_model_path = os.path.join(dir_checkpoint, "best_by_mean_dice.pt")

    if os.path.isfile(best_model_path):
        print(f"Loading best model from: {best_model_path}")
        # 테스트를 위해 새 모델 인스턴스 생성
        test_model = model_lib.SwinSparseCLMOE(
            in_channels=4,
            out_channels=args.num_classes,
            base_dim=args.base_channel,
            expert_count=args.expert_count,
            top_k=args.top_k,
            aux_loss_weight=args.aux_loss_weight,
            drop_path_rate=args.droprate
        ).to(device)
        
        load_checkpoint(best_model_path, test_model, map_location=device)

        test_metrics = evaluate(
            test_model, test_loader, device, args.num_classes,
            roi_size=tuple(args.roi_size),
            is_cl=True,
            sw_batch_size=args.batch_size,
            save_preds=args.save_predictions,  # --- args 값 사용 ---
            output_dir = dir_results,  # --- 경로 전달 ---
            is_test=True
        )



        print("\n--- Final Test Set Results ---")
        for name, value in test_metrics.items():
            if "all" not in name:
                print(
                    f"  {name.replace('_', ' ').title():<20}: {value if not isinstance(value, np.ndarray) else np.round(value, 4)}"
                )

        if use_wandb:
            test_log_data = {}
            for name, value in test_metrics.items():
                if isinstance(value, np.ndarray) and "all" not in name:
                    for i, v in enumerate(value): test_log_data[f"test/{name}_cls{i}"] = v
                elif not isinstance(value, np.ndarray):
                    test_log_data[f"test/{name}"] = value
            wandb.log(test_log_data)

            # [수정/추가] .npy 파일들을 WandB Artifact로 업로드
            npy_results_path = os.path.join(dir_results, 'npy_results')
            if os.path.exists(npy_results_path):
                artifact = wandb.Artifact(
                    name=f"test_npy_details_{wandb.run.id}",
                    type="evaluation_results"
                )
                artifact.add_dir(npy_results_path)
                wandb.log_artifact(artifact)
                print("All detail .npy files uploaded to WandB Artifacts.")

            # Summary 업데이트
            wandb.summary["final_test_dice_mean"] = test_metrics.get("dice_mean", 0.0)
            # ... (기존 summary 코드 유지) ...
            print("Final test results logged to wandb.")

    else:
        print(f"Could not find the best model at {best_model_path} to run final test.")

    if use_wandb:
        wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
