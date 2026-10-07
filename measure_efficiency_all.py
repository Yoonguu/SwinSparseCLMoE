import os
import re
import socket
import importlib.util
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis


# 1. 환경 및 경로 설정
def get_env_paths():

    '''
        paths = {
        "BASE_MODEL_DIR": r'D:\Yoonguu\Papers\To write\SwinSparseCLMoE\Models',
        "DATA_DIR_BRATS21": r'E:\Brain\BraTS2021\TrainingData',
        "DATA_DIR_BRATS24": r'E:\Brain\BRATS2024\BraTS-GLI',
        "TEST_LIST_DIR": r"D:\Yoonguu\PycharmProjects\TumorSeg\SwinUNETR\YG\postprocessing\Test_list\mega\From_TrainVT"
    }

    # 서버별 경로 설정
    if 'mega' in hostname:
        paths["BASE_MODEL_DIR"] = '/home/mega/Desktop/Data1/Code/YG/TumorSeg/SwinUNETR/YG/postprocessing/Models'
        paths["DATA_DIR_BRATS21"] = '/home/mega/Desktop/Data1/Dataset/Brain/BraTS2021/TrainingData'
        paths["DATA_DIR_BRATS24"] = '/home/mega/Desktop/Data1/Dataset/Brain/BRATS2024/BraTS-GLI'
        paths["TEST_LIST_DIR"] = "/home/mega/Desktop/Data1/Code/YG/TumorSeg/SwinUNETR/YG/postprocessing"

    elif 'supermoon' in hostname:
        paths["BASE_MODEL_DIR"] = '/home/leebr/Desktop/YG/Code/TumorSeg/SwinUNETR/YG/postprocessing/Models'
        paths["DATA_DIR_BRATS21"] = '/home/leebr/Desktop/YG/Dataset/Brain/BraTS2021/TrainingData'
        paths["DATA_DIR_BRATS24"] = '/home/leebr/Desktop/YG/Dataset/Brain/BRATS2024/BraTS-GLI'
        paths["TEST_LIST_DIR"] = '/home/leebr/Desktop/YG/Code/TumorSeg/SwinUNETR/YG/postprocessing'

    elif 'gm-a6000' in hostname:
        paths["BASE_MODEL_DIR"] = '/home/gmadmin/DATA2/Song/Code/TumorSeg/SwinSparseCLMoE/postprocessing/Models'
        paths["DATA_DIR_BRATS21"] = '/home/gmadmin/DATA2/Song/Dataset/Brain/BRATS/BraTS2021/TrainingData'
        paths["DATA_DIR_BRATS24"] = '/home/gmadmin/DATA2/Song/Dataset/Brain/BRATS/BraTS2024/BraTS-GLI'
        paths["TEST_LIST_DIR"] = '/home/gmadmin/DATA2/Song/Code/TumorSeg/SwinSparseCLMoE/postprocessing'

    return paths

    :return:
    '''

    hostname = socket.gethostname().lower()
    # paths = {
    #     "BASE_MODEL_DIR": r'D:\Yoonguu\Papers\To write\SwinSparseCLMoE\Models',
    # }

    paths = {
        "BASE_MODEL_DIR": r'D:\Yoonguu\Papers\To write\SwinSparseCLMoE\Models',
        "DATA_DIR_BRATS21": r'E:\Brain\BraTS2021\TrainingData',
        "DATA_DIR_BRATS24": r'E:\Brain\BRATS2024\BraTS-GLI',
        "TEST_LIST_DIR": r"D:\Yoonguu\PycharmProjects\TumorSeg\SwinUNETR\YG\postprocessing\Efficiency"
    }

    if 'mega' in hostname:
        paths["BASE_MODEL_DIR"] = '/home/mega/Desktop/Data1/Code/YG/TumorSeg/SwinUNETR/YG/postprocessing/Models'
        paths["DATA_DIR_BRATS21"] = '/home/mega/Desktop/Data1/Dataset/Brain/BraTS2021/TrainingData'
        paths["DATA_DIR_BRATS24"] = '/home/mega/Desktop/Data1/Dataset/Brain/BRATS2024/BraTS-GLI'
        paths["TEST_LIST_DIR"] = "/home/mega/Desktop/Data1/Code/YG/TumorSeg/SwinUNETR/YG/postprocessing/Efficiency"

    elif 'supermoon' in hostname:
        paths["BASE_MODEL_DIR"] = '/home/leebr/Desktop/YG/Code/TumorSeg/SwinUNETR/YG/postprocessing/Models'
        paths["DATA_DIR_BRATS21"] = '/home/leebr/Desktop/YG/Dataset/Brain/BraTS2021/TrainingData'
        paths["DATA_DIR_BRATS24"] = '/home/leebr/Desktop/YG/Dataset/Brain/BRATS2024/BraTS-GLI'
        paths["TEST_LIST_DIR"] = '/home/leebr/Desktop/YG/Code/TumorSeg/SwinUNETR/YG/postprocessing/Efficiency'
    elif 'gm-a6000' in hostname:
        paths["BASE_MODEL_DIR"] = '/home/gmadmin/DATA2/Song/Code/TumorSeg/SwinSparseCLMoE/postprocessing/Models'
        paths["DATA_DIR_BRATS21"] = '/home/gmadmin/DATA2/Song/Dataset/Brain/BRATS/BraTS2021/TrainingData'
        paths["DATA_DIR_BRATS24"] = '/home/gmadmin/DATA2/Song/Dataset/Brain/BRATS/BraTS2024/BraTS-GLI'
        paths["TEST_LIST_DIR"] = '/home/gmadmin/DATA2/Song/Code/TumorSeg/SwinSparseCLMoE/postprocessing/Efficiency'
    elif 'bmis' in hostname:
        paths["BASE_MODEL_DIR"] = '/home/bmis1/DATA2/YG/Code/TumorSeg/SwinUNETR/YG/postprocessing/Models'
        paths["DATA_DIR_BRATS21"] = '/home/bmis1/DATA2/YG/Dataset/Brain/BraTS2021/TrainingData'
        paths["DATA_DIR_BRATS24"] = 'home/bmis1/DATA2/YG/Dataset/Brain/BRATS2024/BraTS-GLI'
        paths["TEST_LIST_DIR"] = '/home/bmis1/DATA2/YG/Code/TumorSeg/SwinUNETR/YG/postprocessing/Efficiency'


    return paths


ENV_PATHS = get_env_paths()
BASE_MODEL_DIR = ENV_PATHS["BASE_MODEL_DIR"]


TARGET_CONFIG = {
    "nnUNet": {"24": ("BraTS24", "mega_v14.2")},
    "SwinUNeTR": {"24": ("BraTS24_leebr", "mega_v14.2")},
    "MedNeXT": {"24": ("BraTS24", "mega_v14.2")},
    "TransBTS": {"24": ("BraTS24", "mega_v14.2")},
    "U_Mamba": {"24": ("BraTS24", "leebr_v14.3")},
    "SegResNet": {"24": ("BraTS24", "mega_v14.2")},
    "SegMamba": {"24": ("BraTS24", "leebr_v14.2")},
    "SwinSparseCLMoE": {"24": ("BraTS24_24_2_woAux", "mega_v14.2")},
}


def parse_params(folder_name):
    params = {}
    ch_match = re.search(r"ch(\d+)", folder_name)
    drop_match = re.search(r"drop(\d+\.?\d*)", folder_name)
    exp_match = re.search(r"exp(\d+)", folder_name)

    params['base_channel'] = int(ch_match.group(1)) if ch_match else 24
    params['droprate'] = float(drop_match.group(1)) if drop_match else 0.2
    params['expert_count'] = int(exp_match.group(1)) if exp_match else 4
    params['top_k'] = 2
    return params


def get_model_instance_no_weights(model_name, py_path, folder_name, device, num_classes=4):
    spec = importlib.util.spec_from_file_location("dynamic_model", py_path)
    model_lib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_lib)
    p = parse_params(folder_name)

    if 'nnUNet' in model_name:
        model = model_lib.Generic_UNet_Aligned(in_channels=4, num_classes=num_classes,
                                               base_num_features=p['base_channel'])
    elif 'SwinUNeTR' in model_name:
        model = model_lib.SwinUNETR(in_channels=4, out_channels=num_classes, base_dim=p['base_channel'])
    elif 'SegResNet' in model_name:
        model = model_lib.SegResNet(in_channels=4, num_classes=num_classes, base_num_features=p['base_channel'])
    elif 'TransBTS' in model_name:
        model = model_lib.TransBTS(in_channels=4, num_classes=num_classes, base_c=p['base_channel'])
    elif 'U_Mamba' in model_name:
        model = model_lib.UMamba(in_channels=4, num_classes=num_classes, base_c=p['base_channel'])
    elif 'MedNeXT' in model_name:
        model = model_lib.MedNeXt(in_channels=4, num_classes=num_classes, base_c=p['base_channel'])
    elif 'SegMamba' in model_name:
        model = model_lib.SegMamba(in_channels=4, num_classes=num_classes, base_num_features=p['base_channel'])
    elif 'SwinSparseCLMoE' in model_name:
        model = model_lib.SwinSparseCLMOE(in_channels=4, out_channels=num_classes, base_dim=p['base_channel'],
                                          expert_count=p['expert_count'], drop_path_rate=p['droprate'])
    else:
        raise AttributeError(f"Matching class not found for {model_name}")

    model.expert_count = p['expert_count']
    model.top_k = p['top_k']
    return model.to(device)


def calculate_active_params(model, total_params, is_sparse):
    if not is_sparse:
        return total_params

    expert_params = 0
    expert_count = getattr(model, 'expert_count', 4)
    top_k = getattr(model, 'top_k', 2)

    for name, module in model.named_modules():
        if 'expert' in name.lower() and isinstance(module, nn.ModuleList):
            if len(module) > 0:
                expert_params = sum(p.numel() for p in module[0].parameters())
                expert_count = len(module)
                break

    if expert_params == 0:
        for name, module in model.named_modules():
            if name.endswith('experts.0') or name.endswith('expert1') or name.endswith('expert_0'):
                expert_params = sum(p.numel() for p in module.parameters())
                break

    if expert_params > 0:
        inactive_experts = max(0, expert_count - top_k)
        active_params = total_params - (inactive_experts * expert_params)
        return active_params
    else:
        print("  [Warning] Expert modules not dynamically found. Returning Total Params.")
        return total_params


# [추가됨] x2 에러를 해결하기 위한 추론용 래퍼 클래스
class InferenceWrapper(nn.Module):
    def __init__(self, model, is_cl):
        super().__init__()
        self.model = model
        self.is_cl = is_cl

    def forward(self, x):
        if self.is_cl:
            # 대조 학습 모델: 동일한 입력을 두 번(x, x) 전달
            # 반환값 (seg_feat, logits, proj1, proj2, aux_loss) 중 logits(인덱스 1) 반환
            res = self.model(x, x)
            return res[1] if isinstance(res, tuple) else res
        else:
            # 일반 모델
            res = self.model(x)
            return res[0] if isinstance(res, tuple) else res


def measure_efficiency(model, model_name, device, input_shape=(1, 4, 128, 128, 128), num_iterations=100):
    # 1. Parameter Count (동적 계산 적용 - Wrapper 씌우기 전)
    total_params = sum(p.numel() for p in model.parameters())
    is_sparse = "Sparse" in model_name
    is_cl = "CL" in model_name

    active_params = calculate_active_params(model, total_params, is_sparse)

    # 2. x2 에러 방지를 위한 Wrapper 적용
    inference_model = InferenceWrapper(model, is_cl)
    inference_model.eval()

    dummy_input = torch.randn(input_shape).to(device)

    # 3. FLOPs Calculation (fvcore)
    flops_analyzer = FlopCountAnalysis(inference_model, dummy_input)
    flops_analyzer.unsupported_ops_warnings(False)
    flops_analyzer.uncalled_modules_warnings(False)

    total_macs = flops_analyzer.total()
    gflops = (total_macs * 2) / 1e9  # MACs to FLOPs

    # 4. Peak Memory Consumption
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        _ = inference_model(dummy_input)
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # 5. Latency & Throughput
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        for _ in range(10):  # Warm-up
            _ = inference_model(dummy_input)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        for _ in range(num_iterations):
            _ = inference_model(dummy_input)
    end_event.record()
    torch.cuda.synchronize()

    avg_latency_sec = (start_event.elapsed_time(end_event) / 1000.0) / num_iterations
    throughput_fps = 1.0 / avg_latency_sec

    return {
        "Total_Params_M": total_params / 1e6,
        "Active_Params_M": active_params / 1e6,
        "GFLOPs": gflops,
        "Peak_Memory_MB": peak_memory_mb,
        "Latency_sec": avg_latency_sec,
        "Throughput_FPS": throughput_fps
    }


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    output_txt_path = os.path.join(ENV_PATHS["TEST_LIST_DIR"], "efficiency_benchmark_results.txt")


    def log_print(msg, file_obj):
        print(msg)
        file_obj.write(msg + "\n")


    with open(output_txt_path, 'w', encoding='utf-8') as f:
        log_print(f"Benchmarking on: {device} | Input: 1x4x128x128x128 (AMP enabled)", f)
        log_print("-" * 105, f)
        log_print(
            f"{'Model':<18} | {'Params(T/A) M':<15} | {'GFLOPs':<8} | {'Peak Mem(MB)':<12} | {'Latency(s)':<10} | {'FPS':<6}",
            f)
        log_print("-" * 105, f)

        for model_name, versions in TARGET_CONFIG.items():
            try:
                data_folder, result_folder = versions["24"]
                dtype_path = os.path.join(BASE_MODEL_DIR, model_name, data_folder)
                py_files = [file for file in os.listdir(os.path.join(BASE_MODEL_DIR, model_name)) if
                            file.endswith('.py')]

                if not py_files:
                    continue

                location_txt = os.path.join(dtype_path, 'location.txt')
                if os.path.exists(location_txt):
                    with open(location_txt, 'r') as loc_f:
                        folder_name = os.path.basename(os.path.dirname(loc_f.read().strip().replace('\\', '/')))
                else:
                    folder_name = result_folder

                model = get_model_instance_no_weights(model_name, os.path.join(BASE_MODEL_DIR, model_name, py_files[0]),
                                                      folder_name, device)
                results = measure_efficiency(model, model_name, device)

                p_total = f"{results['Total_Params_M']:.2f}"
                p_active = f"{results['Active_Params_M']:.2f}"
                params_str = f"{p_total} / {p_active}"

                log_print(
                    f"{model_name:<18} | {params_str:<15} | {results['GFLOPs']:<8.2f} | {results['Peak_Memory_MB']:<12.1f} | {results['Latency_sec']:<10.4f} | {results['Throughput_FPS']:<6.1f}",
                    f)

                del model
                torch.cuda.empty_cache()

            except Exception as e:
                log_print(f"{model_name:<18} | Error: {e}", f)

        log_print("-" * 105, f)
        log_print(f"\n✅ Results successfully saved to: {output_txt_path}", f)