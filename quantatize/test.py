"""
speed_benchmark.py — Đo FPS inference + Validate (GCNet-official style)

Chạy benchmark (CLI giữ nguyên như cũ — mặc định backend='baseline'):
    python speed_benchmark.py \
        --ckpt ./checkpoints/best.pth \
        --model_variant fan_dwsa \
        --img_h 512 --img_w 1024

Chạy benchmark với backend khác:
    python speed_benchmark.py --ckpt ./checkpoints/best.pth \
        --img_h 512 --img_w 1024 --backend fp16

    python speed_benchmark.py --ckpt ./checkpoints/best.pth \
        --img_h 512 --img_w 1024 --backend compile

    python speed_benchmark.py --ckpt ./checkpoints/best.pth \
        --img_h 512 --img_w 1024 --backend compile_fp16

    python speed_benchmark.py --ckpt ./checkpoints/best.pth \
        --img_h 512 --img_w 1024 --backend torchscript

Chạy validate:
    python speed_benchmark.py \
        --ckpt ./checkpoints/best.pth \
        --val_txt /kaggle/working/val.txt \
        --validate

Chạy cả hai:
    python speed_benchmark.py \
        --ckpt ./checkpoints/best.pth \
        --val_txt /kaggle/working/val.txt \
        --validate --benchmark

--------------------------------------------------------------------------
KIẾN TRÚC BACKEND (để sau này thêm ONNX Runtime / TensorRT / OpenVINO)
--------------------------------------------------------------------------
Mỗi backend là 1 hàm builder: (model, device, img_h, img_w) -> InferenceRunner
InferenceRunner là 1 wrapper callable: runner(x: Tensor) -> Tensor, tự lo việc
convert dtype/device/format bên trong nên benchmark loop KHÔNG cần biết chi
tiết backend đang chạy là gì.

Để thêm backend mới (vd: onnxruntime), chỉ cần:

    @register_backend('onnxruntime')
    def build_onnxruntime_backend(model, device, img_h, img_w, **kwargs):
        import onnxruntime as ort
        # ... export model sang ONNX (hoặc load .onnx có sẵn), tạo InferenceSession
        sess = ort.InferenceSession(...)
        def forward_fn(x):
            return torch.from_numpy(sess.run(None, {'input': x.cpu().numpy()})[0])
        return InferenceRunner('onnxruntime', forward_fn, model=model)

...rồi thêm 'onnxruntime' vào choices của --backend. KHÔNG cần sửa gì trong
benchmark()/main() — toàn bộ phần đo latency/FPS/memory dùng chung.
"""
import argparse
import copy
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.head.segmentation_head import GCNetHead

CLASS_NAMES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]
NUM_CLASSES  = 19
IGNORE_INDEX = 255


# ============================================================
# FUSE CONV + BN
# ============================================================

def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    w     = conv.weight.data
    b     = conv.bias.data if conv.bias is not None else \
            torch.zeros(conv.out_channels, device=w.device)
    scale = bn.weight.data / (bn.running_var + bn.eps).sqrt()
    conv.weight.data = w * scale.reshape(-1, 1, 1, 1)
    conv.bias        = nn.Parameter(bn.bias.data + (b - bn.running_mean) * scale)
    return conv


def fuse_conv_bn(module: nn.Module) -> nn.Module:
    # Recurse vào tất cả children trước
    for child in module.children():
        fuse_conv_bn(child)
    # Fuse các cặp Conv→BN là direct children
    children = list(module.named_children())
    i = 0
    while i < len(children) - 1:
        name_a, mod_a = children[i]
        name_b, mod_b = children[i + 1]
        if isinstance(mod_a, nn.Conv2d) and \
                isinstance(mod_b, (nn.BatchNorm2d, nn.SyncBatchNorm)) and \
                mod_a.out_channels == mod_b.num_features:
            module._modules[name_a] = _fuse_conv_bn(mod_a, mod_b)
            module._modules[name_b] = nn.Identity()
            i += 2
        else:
            i += 1
    return module


# ============================================================
# BUILD MODEL
# ============================================================

class Segmentor(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone    = backbone
        self.decode_head = head

    def forward(self, x):
        return self.decode_head(self.backbone(x))


def build_model(variant: str, ckpt_path: str, device: torch.device,
                deploy: bool = True) -> Segmentor:
    C   = 32
    cfg = dict(
        in_channels=3, channels=C, ppm_channels=128,
        num_blocks_per_stage=[4, 4, [5, 4], [5, 4], [2, 2]],
        align_corners=False, deploy=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
    )
    if variant == 'fan_dwsa':
        from model.backbone.model import GCNet
        cfg['dwsa_reduction'] = 8
    elif variant == 'fan_only':
        from model.backbone.fan import GCNet
    else:
        from model.backbone.dwsa import GCNet
        cfg['dwsa_reduction'] = 8

    model = Segmentor(
        GCNet(**cfg),
        GCNetHead(
            in_channels=C*4, channels=64, num_classes=NUM_CLASSES,
            align_corners=False, dropout_ratio=0.0, ignore_index=IGNORE_INDEX,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=dict(type='ReLU', inplace=True),
        )
    )
    ck    = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = (ck.get('model') or ck.get('model_state_dict') or
             ck.get('state_dict') or ck)
    model.load_state_dict(state, strict=False)
    recorded = ck.get('best_miou', '?')
    print(f"  Loaded {variant} | recorded mIoU: {recorded}")

    if deploy:
        model.backbone.switch_to_deploy()
        print(f"  switch_to_deploy applied (reparam branches fused)")

    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_conv   = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    print(f"  params: {n_params:.2f}M | Conv2d layers: {n_conv}")
    return model, recorded


# ============================================================
# BACKEND REGISTRY
# ============================================================
# Đây là điểm mở rộng DUY NHẤT cần chạm vào khi thêm ONNX Runtime / TensorRT /
# OpenVINO sau này — xem docstring đầu file để biết cách thêm 1 backend mới.

BACKEND_REGISTRY = {}


def register_backend(name):
    def deco(fn):
        BACKEND_REGISTRY[name] = fn
        return fn
    return deco


class InferenceRunner:
    """Wrapper thống nhất cho mọi backend: runner(x) -> output tensor (CPU, float32).

    benchmark()/validate() chỉ cần gọi runner(x) — không quan tâm backend bên
    dưới là PyTorch thuần, TorchScript, torch.compile, hay (sau này) ONNX
    Runtime/TensorRT/OpenVINO.
    """
    def __init__(self, name: str, forward_fn, model: nn.Module = None):
        self.name = name
        self.forward_fn = forward_fn
        self.model = model  # giữ lại reference model gốc (để đếm params, v.v.)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_fn(x)


@register_backend('baseline')
def build_baseline_backend(model, device, img_h, img_w, **kwargs):
    """PyTorch eager mode thuần — baseline để so sánh mọi backend khác."""
    def forward_fn(x):
        with torch.no_grad():
            return model(x)
    return InferenceRunner('baseline', forward_fn, model=model)


@register_backend('fp16')
def build_fp16_backend(model, device, img_h, img_w, **kwargs):
    """Half-precision (FP16) — tăng tốc đáng kể trên GPU có Tensor Cores
    (Jetson Orin/Xavier, RTX...). Trên CPU, FP16 thường KHÔNG được tối ưu
    (thiếu kernel FP16 hiệu quả) nên có thể chậm hơn hoặc lỗi — script sẽ
    cảnh báo rõ trong trường hợp đó.
    """
    if device.type != 'cuda':
        print("  [!] FP16 backend chạy trên CPU thường KHÔNG có lợi về tốc độ "
              "(thiếu kernel FP16 tối ưu) — khuyến nghị test FP16 trên GPU/Jetson thật.")
    model_fp16 = model.half()

    # FIX: sau .half(), BatchNorm cũng bị chuyển FP16 — running_mean/running_var
    # tích luỹ qua nhiều batch dễ mất precision ở FP16 (giá trị nhỏ bị underflow/
    # rounding), làm giảm accuracy so với FP32 gốc. Best-practice chuẩn (giống
    # torch.cuda.amp): giữ riêng BatchNorm ở FP32, chỉ Conv/Linear chạy FP16.
    # PyTorch tự động upcast/downcast qua lại giữa 2 dtype ở input/output của
    # BN layer nên không cần sửa gì thêm trong forward.
    for m in model_fp16.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.InstanceNorm2d)):
            m.float()

    def forward_fn(x):
        with torch.no_grad():
            return model_fp16(x.half())
    return InferenceRunner('fp16', forward_fn, model=model_fp16)


@register_backend('compile')
def build_compile_backend(model, device, img_h, img_w, **kwargs):
    """torch.compile (TorchInductor) — tự động fuse kernel, giảm overhead
    Python dispatch. Cần PyTorch >= 2.0. Lần gọi đầu tiên sẽ có compile
    overhead (không tính vào latency vì đã warmup trước khi đo).

    mode='reduce-overhead' dùng CUDA Graphs — giảm mạnh Python/launch overhead
    cho inference lặp lại nhiều lần (đúng kịch bản benchmark ở đây), nhưng
    CHỈ có ý nghĩa trên CUDA (CPU không có CUDA Graphs) và yêu cầu input shape
    cố định qua các lần gọi — khớp với chiến lược fixed-shape của cả pipeline.
    """
    compile_mode = 'reduce-overhead' if device.type == 'cuda' else None
    print(f"  torch.compile mode={compile_mode!r}"
          + ("" if compile_mode else " (CPU — reduce-overhead chỉ hỗ trợ CUDA)"))
    compiled = torch.compile(model, mode=compile_mode)

    def forward_fn(x):
        with torch.no_grad():
            return compiled(x)
    return InferenceRunner('compile', forward_fn, model=model)


@register_backend('compile_fp16')
def build_compile_fp16_backend(model, device, img_h, img_w, **kwargs):
    """Kết hợp torch.compile + FP16 — thường cho tốc độ tốt nhất trên GPU
    hiện đại hỗ trợ cả Tensor Cores lẫn kernel fusion.
    """
    if device.type != 'cuda':
        print("  [!] compile_fp16 trên CPU có thể không ổn định/không có lợi — "
              "khuyến nghị chạy trên GPU/Jetson thật.")
    model_fp16 = model.half()
    for m in model_fp16.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.InstanceNorm2d)):
            m.float()
    compile_mode = 'reduce-overhead' if device.type == 'cuda' else None
    compiled = torch.compile(model_fp16, mode=compile_mode)

    def forward_fn(x):
        with torch.no_grad():
            return compiled(x.half())
    return InferenceRunner('compile_fp16', forward_fn, model=model_fp16)


@register_backend('torchscript')
def build_torchscript_backend(model, device, img_h, img_w, **kwargs):
    """TorchScript (trace) — đóng băng graph, loại bỏ Python overhead, dễ
    deploy sang môi trường không có Python (C++ libtorch) — hữu ích cho
    Embedded Linux / production serving.
    """
    example = torch.randn(1, 3, img_h, img_w, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    traced = torch.jit.freeze(traced) if not traced.training else traced

    def forward_fn(x):
        with torch.no_grad():
            return traced(x)
    return InferenceRunner('torchscript', forward_fn, model=model)


def _export_onnx(model, onnx_path, img_h, img_w, opset=17, half=False):
    """Export model (đã deploy=True, reparam fused) sang ONNX — dùng chung
    cho cả 3 backend onnxruntime/tensorrt/openvino để tránh lặp code.
    Luôn export trên CPU cho ổn định, sau đó trả model về đúng device gốc.

    half=True: export ONNX với toàn bộ tensor ở FP16 — CẦN THIẾT cho TensorRT
    >= 11.x khi muốn build engine FP16, vì BuilderFlag.FP16 đã bị xoá (network
    giờ "strongly typed", precision lấy trực tiếp từ dtype của chính ONNX graph
    thay vì set qua builder config).
    """
    orig_device = next(model.parameters()).device
    os.makedirs(os.path.dirname(onnx_path) or '.', exist_ok=True)
    print(f"  Exporting ONNX -> {onnx_path} (opset={opset}, half={half})...")

    if half:
        # deepcopy để không làm hỏng model gốc (vẫn cần dùng FP32 cho các
        # backend khác trong cùng 1 lần chạy nếu có)
        model_export = copy.deepcopy(model).to('cpu').eval().half()
        dummy = torch.randn(1, 3, img_h, img_w).half()
    else:
        model_export = model.to('cpu').eval()
        dummy = torch.randn(1, 3, img_h, img_w)

    torch.onnx.export(
        model_export, dummy, onnx_path,
        input_names=['input'], output_names=['output'],
        opset_version=opset, do_constant_folding=True,
        dynamo=False,  # TorchScript-based exporter — ổn định, không cần onnxscript
    )
    model.to(orig_device)
    print(f"  ✅ Exported: {onnx_path} ({os.path.getsize(onnx_path)/1024**2:.2f} MB)")


def _get_or_export_onnx(model, img_h, img_w, onnx_path=None, force_rebuild=False,
                         half=False, **_):
    if onnx_path is None:
        suffix = '_fp16' if half else ''
        onnx_path = f'checkpoints/gcnet_{img_h}x{img_w}{suffix}.onnx'
    if force_rebuild or not os.path.exists(onnx_path):
        _export_onnx(model, onnx_path, img_h, img_w, half=half)
    else:
        print(f"  Dùng ONNX có sẵn: {onnx_path} (thêm --force_rebuild để export lại)")
    return onnx_path


class _RandomOrDirCalibrationReader:
    """Calibration data reader cho quantize_static — đọc ảnh thật từ calib_dir
    nếu có, nếu không thì fallback random tensor (CHỈ để demo/test pipeline,
    KHÔNG dùng random data cho INT8 production — accuracy sẽ sai lệch nhiều
    vì range hoạt động không phản ánh đúng phân phối dữ liệu thật).
    """
    def __init__(self, input_name, img_h, img_w, calib_dir=None, n_samples=100):
        self.input_name = input_name
        self.idx = 0
        if calib_dir and os.path.isdir(calib_dir):
            import glob
            import cv2
            paths = sorted(glob.glob(os.path.join(calib_dir, '*')))[:n_samples]
            self.samples = []
            for pth in paths:
                img = cv2.imread(pth)
                if img is None:
                    continue
                img = cv2.resize(img, (img_w, img_h))
                img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
                self.samples.append(img[None])
            print(f"  Đọc {len(self.samples)} ảnh calibration từ {calib_dir}")
        else:
            print(f"  [!] Không có --calib_dir hợp lệ -> dùng {n_samples} random "
                  f"tensor để calibrate (CHỈ DEMO — INT8 build từ random data "
                  f"KHÔNG phản ánh đúng phân phối dữ liệu thật, đừng dùng số liệu "
                  f"accuracy này cho production).")
            self.samples = [np.random.rand(1, 3, img_h, img_w).astype(np.float32)
                             for _ in range(n_samples)]

    def get_next(self):
        if self.idx >= len(self.samples):
            return None
        item = {self.input_name: self.samples[self.idx]}
        self.idx += 1
        return item

    def rewind(self):
        self.idx = 0


def _get_or_build_qdq_onnx(fp32_onnx_path, img_h, img_w, calib_dir=None,
                            force_rebuild=False):
    """Tạo (hoặc dùng cache) ONNX dạng QDQ (QuantizeLinear/DequantizeLinear) từ
    ONNX FP32 — ĐÂY LÀ FORMAT TensorRT >= 11.x CẦN để build engine INT8 thật
    (network strongly-typed lấy precision trực tiếp từ node Q/DQ trong graph,
    không còn implicit calibrator qua BuilderFlag.INT8 nữa).
    """
    from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationMethod
    import onnxruntime as ort

    qdq_path = fp32_onnx_path.replace('.onnx', '_qdq.onnx')
    if not force_rebuild and os.path.exists(qdq_path):
        print(f"  Dùng ONNX QDQ có sẵn: {qdq_path} (thêm --force_rebuild để tạo lại)")
        return qdq_path

    sess = ort.InferenceSession(fp32_onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    reader = _RandomOrDirCalibrationReader(input_name, img_h, img_w, calib_dir)

    print(f"  Đang tạo ONNX QDQ (static quantization) -> {qdq_path} ...")
    quantize_static(
        model_input=fp32_onnx_path,
        model_output=qdq_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,   # BẮT BUỘC: QDQ format để TensorRT parse đúng
        calibrate_method=CalibrationMethod.MinMax,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        extra_options={
            # FIX: mặc định onnxruntime quantize bias thành Int32 kèm node
            # DequantizeLinear riêng — TensorRT KHÔNG chấp nhận DequantizeLinear
            # trên input Int32 (chỉ nhận INT8/FP8/FP4/INT4), gây lỗi parse
            # ("ITensor::getDimensions ... must have type FP8, FP4, Int4, or Int8").
            # Giữ bias ở FP32 (không quantize) — TensorRT vẫn cộng bias bình
            # thường sau Conv INT8, gần như không ảnh hưởng tốc độ vì bias-add
            # chỉ là 1 phép cộng nhỏ so với toàn bộ phép nhân ma trận Conv.
            'QuantizeBias': False,
        },
    )
    print(f"  ✅ ONNX QDQ saved: {qdq_path}")
    return qdq_path


@register_backend('onnxruntime')
def build_onnxruntime_backend(model, device, img_h, img_w,
                               onnx_path=None, force_rebuild=False, **kwargs):
    """ONNX Runtime — tự chọn CUDAExecutionProvider nếu có GPU, fallback CPU.

    Trên máy bạn (RTX 2050, ONNX Runtime 1.27 build có CUDA): sẽ tự dùng
    CUDAExecutionProvider + IOBinding (zero-copy GPU->ORT->GPU, không phải
    đi vòng qua CPU numpy như cách gọi sess.run() thông thường).
    """
    import onnxruntime as ort

    onnx_path = _get_or_export_onnx(model, img_h, img_w, onnx_path, force_rebuild)

    available = ort.get_available_providers()
    use_cuda = (device.type == 'cuda' and 'CUDAExecutionProvider' in available)
    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_cuda
                 else ['CPUExecutionProvider'])
    print(f"  ONNX Runtime available providers: {available}")
    print(f"  ONNX Runtime dùng providers: {providers}")

    sess = ort.InferenceSession(onnx_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    if use_cuda:
        # --- IOBinding: input/output ở thẳng trên GPU, không copy qua CPU ---
        import torch.utils.dlpack as dlpack
        io_binding = sess.io_binding()
        print("  Dùng IOBinding (zero-copy GPU input/output)")

        def forward_fn(x):
            x = x.to(device='cuda', dtype=torch.float32).contiguous()
            io_binding.bind_input(
                name=input_name, device_type='cuda', device_id=0,
                element_type=np.float32, shape=tuple(x.shape),
                buffer_ptr=x.data_ptr(),
            )
            io_binding.bind_output(output_name, device_type='cuda', device_id=0)
            sess.run_with_iobinding(io_binding)
            ort_out = io_binding.get_outputs()[0]
            # Chuyển OrtValue -> torch tensor qua DLPack, KHÔNG copy dữ liệu,
            # kết quả vẫn nằm trên GPU (đúng ý #1 trong review: tránh .cpu()
            # thừa làm sai lệch số liệu latency thuần).
            return dlpack.from_dlpack(ort_out.to_dlpack())
    else:
        # CPU: numpy path bình thường, không cần IOBinding (không có copy
        # GPU<->CPU để tối ưu vì bản thân đã chạy trên CPU).
        def forward_fn(x):
            x_np = x.detach().cpu().numpy()
            out_np = sess.run(None, {input_name: x_np})[0]
            return torch.from_numpy(out_np)

    return InferenceRunner('onnxruntime', forward_fn, model=model)


@register_backend('openvino')
def build_openvino_backend(model, device, img_h, img_w,
                            onnx_path=None, force_rebuild=False,
                            ov_device='AUTO', ov_precision='FP16', **kwargs):
    """OpenVINO — dùng cho Intel CPU/iGPU/NPU/Core Ultra.

    ov_device: 'CPU' / 'GPU' (iGPU Intel) / 'NPU' / 'AUTO'
    ov_precision: 'FP16' (khuyến nghị, giảm ~50% size) hoặc 'FP32'
    """
    import openvino as ov

    onnx_path = _get_or_export_onnx(model, img_h, img_w, onnx_path, force_rebuild)

    core = ov.Core()
    print(f"  OpenVINO devices khả dụng: {core.available_devices}")
    ov_model = ov.convert_model(onnx_path)

    ir_path = onnx_path.replace('.onnx', f'_{ov_precision.lower()}.xml')
    ov.save_model(ov_model, ir_path, compress_to_fp16=(ov_precision.upper() == 'FP16'))
    print(f"  ✅ OpenVINO IR saved: {ir_path}")

    compiled = core.compile_model(ov_model, ov_device)
    infer_request = compiled.create_infer_request()

    def forward_fn(x):
        x_np = x.detach().cpu().numpy()
        result = infer_request.infer({0: x_np})
        out = result[compiled.output(0)]
        return torch.from_numpy(out)

    return InferenceRunner(f'openvino[{ov_device}/{ov_precision}]', forward_fn, model=model)


# --- TensorRT: dùng API "bindingless" (execute_async_v3 + set_tensor_address) ---
# đúng chuẩn TensorRT >= 8.5, khớp với TensorRT 11.1 trên máy bạn. KHÔNG tương
# thích với code TensorRT 8.x kiểu cũ (execute_async_v2 + bindings list).

_TRT_DTYPE_TO_TORCH = None  # lazy-init trong hàm vì cần import tensorrt trước


def _build_tensorrt_engine(onnx_path, engine_path, precision='fp16', workspace_gb=4):
    import tensorrt as trt

    trt_major = int(trt.__version__.split('.')[0])
    print(f"  TensorRT version: {trt.__version__} (major={trt_major})")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    print(f"  Parsing ONNX -> TensorRT network: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  [TRT parse error] {parser.get_error(i)}")
            raise RuntimeError("Parse ONNX sang TensorRT network that bai")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    if trt_major >= 11:
        # TensorRT >= 11.0: TOAN BO BuilderFlag.FP16/INT8/BF16/FP8/INT4 (weak
        # typing) da bi XOA (xem NVIDIA "Migrating from TensorRT 10.x to 11.x").
        # Network gio luon "strongly typed" - precision cua tung tensor lay
        # TRUC TIEP tu dtype khai bao trong chinh do thi ONNX, khong con set
        # qua builder config nua. Do do:
        #   - fp16: ONNX dua vao PHAI LA ONNX FP16 that (xem _export_onnx(half=True)
        #     o build_tensorrt_backend) - khong set flag gi them o day.
        #   - int8: ONNX phai co san node QuantizeLinear/DequantizeLinear (explicit
        #     quantization/QDQ graph). IInt8Calibrator kieu implicit cu DA BI XOA.
        #     Dung scripts/04_quantize_ptq.py (--mode static) de tao ONNX QDQ roi
        #     truyen vao qua --onnx_path, KHONG dung calibrator o day nua.
        if precision == 'fp16':
            print("  [TensorRT >=11] Strongly-typed network: engine se FP16 neu (va "
                  "chi neu) ONNX input da la FP16 - khong dung BuilderFlag.FP16 nua.")
        elif precision == 'int8':
            print("  [TensorRT >=11] INT8: ONNX truyền vào đã ở dạng QDQ "
                  "(QuantizeLinear/DequantizeLinear, tự động tạo ở build_tensorrt_backend) "
                  "-> engine sẽ dùng kernel INT8 thật cho các layer đã quantize.")
        # fp32: khong can lam gi them, ONNX FP32 -> engine FP32 mac dinh.
    else:
        # TensorRT <= 10.x: API cu, van con BuilderFlag
        if precision == 'fp16':
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("  FP16 mode: bat (GPU ho tro FP16 toc do cao)")
            else:
                print("  [!] GPU khong bao ho tro FP16 nhanh - van build FP32.")
        elif precision == 'int8':
            config.set_flag(trt.BuilderFlag.INT8)
            print("  [!] INT8 build KHONG co calibrator rieng o day (dung default) - "
                  "CHI de demo toc do, PHAI calibrate bang du lieu that truoc khi dung "
                  "production (xem scripts/04_quantize_ptq.py de tham khao cach calibrate).")

    print("  Building TensorRT engine (co the mat vai phut)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Build TensorRT engine that bai (build_serialized_network tra None)")

    os.makedirs(os.path.dirname(engine_path) or '.', exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    print(f"  Da luu TensorRT engine: {engine_path} "
          f"({os.path.getsize(engine_path)/1024**2:.2f} MB)")


@register_backend('tensorrt')
def build_tensorrt_backend(model, device, img_h, img_w,
                            onnx_path=None, trt_engine_path=None,
                            trt_precision='fp16', trt_workspace_gb=4,
                            force_rebuild=False, calib_dir=None, **kwargs):
    """TensorRT native API — nhanh nhất trên NVIDIA GPU/Jetson, bắt buộc CUDA.

    Input size PHẢI CỐ ĐỊNH khi build engine (không dùng dynamic_axes) để đạt
    tốc độ tối ưu nhất — đây cũng là khuyến nghị chuẩn cho deploy Edge AI.
    """
    if device.type != 'cuda':
        raise RuntimeError(
            "Backend 'tensorrt' yêu cầu CUDA device (GPU) — không chạy được trên CPU. "
            "Chạy lại với máy có GPU (vd: RTX 2050) hoặc trên Jetson.")

    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError(
            "Chưa cài package `tensorrt`. Trên máy có RTX/GPU thường: cài qua "
            "NVIDIA pip index đúng bản CUDA (vd: `pip install tensorrt` sau khi "
            "đã cài CUDA Toolkit + cuDNN khớp version), hoặc trên Jetson TensorRT "
            "đã có sẵn trong JetPack (dùng python3 hệ thống, không phải venv riêng "
            "trừ khi symlink lại site-packages tensorrt)."
        ) from e

    trt_major = int(trt.__version__.split('.')[0])

    # FIX: TensorRT >= 11.x đã xoá BuilderFlag.FP16 — muốn engine FP16 phải
    # đưa vào 1 ONNX FP16 thật (strongly-typed network lấy precision từ dtype
    # của chính đồ thị ONNX). Với TensorRT <= 10.x, vẫn export ONNX FP32 bình
    # thường và set flag như cũ.
    need_fp16_onnx = (trt_major >= 11 and trt_precision == 'fp16')
    if need_fp16_onnx:
        print(f"  TensorRT {trt.__version__} (>=11): export ONNX dạng FP16 để build "
              f"engine FP16 (BuilderFlag.FP16 đã bị xoá khỏi TensorRT 11.x).")

    onnx_path = _get_or_export_onnx(model, img_h, img_w, onnx_path, force_rebuild,
                                     half=need_fp16_onnx)

    # FIX: TensorRT >= 11.x cần ONNX dạng QDQ (QuantizeLinear/DequantizeLinear)
    # để build engine INT8 THẬT — implicit BuilderFlag.INT8 + calibrator kiểu cũ
    # đã bị xoá. Trước đây code chỉ IN CẢNH BÁO rồi build thẳng từ ONNX FP32
    # thường -> engine build "thành công" nhưng KHÔNG có kernel INT8 nào cả
    # (chạy full-precision, tên file gây hiểu lầm là INT8). Giờ tự động tạo
    # ONNX QDQ (static quantization) trước khi build, giống hệt cách
    # scripts/04_quantize_ptq.py --mode static làm, nhưng tích hợp thẳng vào
    # đây để chạy 1 lệnh duy nhất là đủ.
    build_onnx_path = onnx_path
    if trt_precision == 'int8' and trt_major >= 11:
        build_onnx_path = _get_or_build_qdq_onnx(
            onnx_path, img_h, img_w, calib_dir=calib_dir, force_rebuild=force_rebuild)

    engine_path = trt_engine_path or onnx_path.replace('.onnx', f'_{trt_precision}.engine')

    if force_rebuild or not os.path.exists(engine_path):
        _build_tensorrt_engine(build_onnx_path, engine_path, trt_precision, trt_workspace_gb)
    else:
        print(f"  Dùng TensorRT engine có sẵn: {engine_path} "
              f"(thêm --force_rebuild để build lại)")

    trt_logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(trt_logger)
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Xác định tên input/output tensor theo IO mode (không giả định thứ tự cố định)
    input_name, output_name = None, None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_name = name
        else:
            output_name = name
    assert input_name and output_name, "Không tìm thấy input/output tensor trong engine"

    dtype_map = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.BOOL: torch.bool,
    }
    in_dtype = dtype_map[engine.get_tensor_dtype(input_name)]
    out_dtype = dtype_map[engine.get_tensor_dtype(output_name)]

    stream = torch.cuda.Stream()

    def forward_fn(x):
        x = x.to(device='cuda', dtype=in_dtype).contiguous()
        context.set_input_shape(input_name, tuple(x.shape))
        out_shape = tuple(context.get_tensor_shape(output_name))
        out = torch.empty(out_shape, dtype=out_dtype, device='cuda')

        context.set_tensor_address(input_name, x.data_ptr())
        context.set_tensor_address(output_name, out.data_ptr())
        with torch.cuda.stream(stream):
            context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        # KHÔNG copy về CPU ở đây — .cpu() sẽ cộng thêm PCIe transfer time vào
        # số liệu benchmark, làm sai lệch latency "pure inference". Giữ output
        # trên GPU; benchmark() chỉ cần torch.cuda.synchronize() là đủ để đo
        # đúng thời gian tính toán. Nếu cần accuracy-check (so sánh với PyTorch/
        # ONNX Runtime), gọi runner(x).cpu().numpy() ở nơi gọi, không phải ở đây.
        return out

    return InferenceRunner(f'tensorrt[{trt_precision}]', forward_fn, model=model)


def _not_implemented_backend(name, hint):
    @register_backend(name)
    def _stub(model, device, img_h, img_w, **kwargs):
        raise NotImplementedError(
            f"Backend '{name}' chưa được tích hợp trực tiếp vào speed_benchmark.py.\n"
            f"  -> {hint}"
        )
    return _stub


# Đã implement đầy đủ ONNX Runtime / TensorRT / OpenVINO ở trên (không còn là
# placeholder). Giữ lại register_backend cho backend chưa hỗ trợ khác nếu cần
# mở rộng thêm sau này (vd: 'ncnn', 'tflite'...).


def build_runner(backend: str, model: nn.Module, device: torch.device,
                  img_h: int, img_w: int, **kwargs) -> InferenceRunner:
    if backend not in BACKEND_REGISTRY:
        raise ValueError(f"Backend '{backend}' không tồn tại. "
                          f"Các backend khả dụng: {list(BACKEND_REGISTRY.keys())}")
    return BACKEND_REGISTRY[backend](model, device, img_h, img_w, **kwargs)


# ============================================================
# VALIDATE — GCNet official methodology (khớp iou_metric.py)
# ============================================================

def _update_accum(pred, target, ti, tu, tp, tl):
    """
    Cập nhật 4 accumulators cho một batch.
    Khớp với intersect_and_union() của GCNet iou_metric.py:
      area_intersect = histc(pred[pred==label])
      area_pred      = histc(pred)
      area_label     = histc(label)
      area_union     = area_pred + area_label - area_intersect
    """
    mask = (target != IGNORE_INDEX) & (target >= 0) & (target < NUM_CLASSES)
    p    = pred[mask].astype(np.int64)
    t    = target[mask].astype(np.int64)
    intersect = p[p == t]
    ai = np.bincount(intersect, minlength=NUM_CLASSES)
    ap = np.bincount(p,         minlength=NUM_CLASSES)
    al = np.bincount(t,         minlength=NUM_CLASSES)
    ti += ai;  tu += ap + al - ai;  tp += ap;  tl += al


def _compute_metrics(ti, tu, tp, tl):
    """Tính metrics từ accumulated totals — khớp total_area_to_metrics()."""
    present = tl > 0
    iou     = ti / (tu + 1e-10)
    acc     = ti / (tl + 1e-10)
    dice    = 2 * ti / (tp + tl + 1e-10)
    aacc    = float(ti[present].sum() / (tl[present].sum() + 1e-10))
    return {
        'aacc'          : aacc,
        'miou'          : float(np.nanmean(iou[present])),
        'macc'          : float(np.nanmean(acc[present])),
        'mdice'         : float(np.nanmean(dice[present])),
        'per_class_iou' : iou,
        'per_class_acc' : acc,
        'per_class_dice': dice,
        'present'       : present,
    }


@torch.no_grad()
def validate(model, val_txt, img_h, img_w, batch_size,
             num_workers, device, use_amp, recorded):
    from data.custom import CityscapesDataset, get_val_transforms

    val_ds = CityscapesDataset(
        txt_file=val_txt,
        transforms=get_val_transforms(img_size=(img_h, img_w)),
        img_size=(img_h, img_w),
        label_mapping='train_id',
        dataset_type='foggy',
    )
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == 'cuda'),
        drop_last=False,
    )
    print(f"\n  Val: {len(val_ds):,} samples | {len(loader)} batches\n")

    ti = np.zeros(NUM_CLASSES, dtype=np.int64)  # total_intersect
    tu = np.zeros(NUM_CLASSES, dtype=np.int64)  # total_union
    tp = np.zeros(NUM_CLASSES, dtype=np.int64)  # total_pred
    tl = np.zeros(NUM_CLASSES, dtype=np.int64)  # total_label
    total_loss = 0.0
    ce_fn      = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    model.eval()
    pbar = tqdm(loader, desc="Validating", ncols=90)
    for imgs, masks in pbar:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()
        if masks.dim() == 4:
            masks = masks.squeeze(1)
        with autocast(device_type='cuda', enabled=use_amp):
            logits = model(imgs)
            logits = F.interpolate(logits, size=masks.shape[-2:],
                                   mode='bilinear', align_corners=False)
            loss   = ce_fn(logits, masks)
        total_loss += loss.item()
        _update_accum(logits.argmax(1).cpu().numpy(),
                      masks.cpu().numpy(), ti, tu, tp, tl)
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    m = _compute_metrics(ti, tu, tp, tl)
    m['loss'] = total_loss / len(loader)

    # Print
    print(f"\n{'='*65}")
    print(f"  VALIDATION RESULTS  (GCNet-official methodology)")
    print(f"{'='*65}")
    print(f"  aAcc:   {m['aacc']:.4f}   (overall pixel accuracy)")
    print(f"  mIoU:   {m['miou']:.4f}")
    print(f"  mAcc:   {m['macc']:.4f}   (mean per-class recall)")
    print(f"  mDice:  {m['mdice']:.4f}")
    print(f"  Loss:   {m['loss']:.4f}   (deploy: không so với training log)")
    print(f"{'='*65}")
    print(f"\n  {'Class':<16} {'IoU':>6}  {'Acc':>6}  {'Dice':>6}  Bar")
    print(f"  {'─'*60}")
    for name, i, a, d, p in zip(CLASS_NAMES,
                                 m['per_class_iou'],
                                 m['per_class_acc'],
                                 m['per_class_dice'],
                                 m['present']):
        bar  = '█' * int(i * 20)
        mark = ' ⚠️ ' if i < 0.40 else (' ★' if i > 0.75 else '')
        note = '' if p else ' (no GT)'
        print(f"  {name:<16} {i:>6.4f}  {a:>6.4f}  {d:>6.4f}  {bar}{mark}{note}")

    low = [n for n, v, p in zip(CLASS_NAMES, m['per_class_iou'], m['present'])
           if v < 0.40 and p]
    if low:
        print(f"\n  ⚠️  LOW IoU (<0.40): {low}")
    print(f"{'='*65}")

    if recorded and recorded != '?':
        diff = abs(m['miou'] - float(recorded))
        ok   = '✅' if diff < 0.001 else '⚠️ '
        print(f"\n  {ok} mIoU: {m['miou']:.4f} vs recorded {float(recorded):.4f}"
              f"  (diff={diff:.4f})")
    print()
    return m


# ============================================================
# BENCHMARK
# ============================================================

def _run_iters(runner, inp, n):
    for _ in range(n):
        runner(inp)


def benchmark(runner: InferenceRunner, img_h, img_w, device,
              n_warmup=50, n_repeat=3, target_sec=6.0):
    is_cuda = (device.type == 'cuda')
    torch.backends.cudnn.enabled   = True
    torch.backends.cudnn.benchmark = True

    inp = torch.randn(1, 3, img_h, img_w, device=device)

    print(f"  Warmup {n_warmup} iters...")
    for _ in range(n_warmup):
        runner(inp)
    if is_cuda:
        torch.cuda.synchronize(device)

    print(f"  Auto-calibrating (target {target_sec}s/run)...")
    n_iters = 100
    while True:
        if is_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _run_iters(runner, inp, n_iters)
        if is_cuda:
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        if elapsed >= 1.0:
            break
        n_iters *= 2
    n_iters = max(int(n_iters / elapsed * target_sec), 100)
    print(f"  n_iters/run: {n_iters}")

    fps_list, lat_list = [], []
    for r in range(n_repeat):
        if is_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _run_iters(runner, inp, n_iters)
        if is_cuda:
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        fps = n_iters / elapsed
        lat = elapsed / n_iters * 1000
        fps_list.append(fps);  lat_list.append(lat)
        print(f"    run {r+1}/{n_repeat}: {fps:.1f} FPS  {lat:.2f} ms")

    mem_mb = None
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        runner(inp)
        torch.cuda.synchronize(device)
        mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2

    params_m = None
    if runner.model is not None:
        try:
            params_m = sum(p.numel() for p in runner.model.parameters()) / 1e6
        except Exception:
            params_m = None

    fps_arr = np.array(fps_list)
    lat_arr = np.array(lat_list)
    return {
        'fps_median': float(np.median(fps_arr)),
        'fps_mean'  : float(fps_arr.mean()),
        'fps_std'   : float(fps_arr.std()),
        'ms_median' : float(np.median(lat_arr)),
        'ms_mean'   : float(lat_arr.mean()),
        'ms_std'    : float(lat_arr.std()),
        'mem_mb'    : mem_mb,
        'params_m'  : params_m,
        'n_iters'   : n_iters,
        'n_repeat'  : n_repeat,
    }


def print_benchmark(r, variant, backend, img_h, img_w, gpu_name):
    print(f"\n{'='*55}")
    print(f"  INFERENCE SPEED  —  {variant} | backend={backend}  (deploy+fuse)")
    print(f"{'='*55}")
    print(f"  Device:        {gpu_name}")
    print(f"  Input:         1 × 3 × {img_h} × {img_w}")
    print(f"  Iters/run:     {r['n_iters']}  ×  {r['n_repeat']} runs")
    print(f"  {'─'*43}")
    print(f"  FPS (median):  {r['fps_median']:>8.1f}")
    print(f"  FPS (mean):    {r['fps_mean']:>8.1f}  ±  {r['fps_std']:.1f}")
    print(f"  {'─'*43}")
    print(f"  ms  (median):  {r['ms_median']:>8.2f} ms")
    print(f"  ms  (mean):    {r['ms_mean']:>8.2f}  ±  {r['ms_std']:.2f} ms")
    print(f"  {'─'*43}")
    if r['mem_mb'] is not None:
        print(f"  GPU mem peak:  {r['mem_mb']:>8.1f} MB  (1 forward pass)")
    else:
        print(f"  GPU mem peak:  n/a (CPU backend)")
    if r['params_m'] is not None:
        print(f"  Params:        {r['params_m']:>8.2f} M")
    print(f"{'='*55}\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="GCNet v3 — Benchmark + Validate")
    parser.add_argument('--ckpt',          required=True)
    parser.add_argument('--model_variant', default='fan_dwsa',
                        choices=['fan_dwsa', 'fan_only', 'dwsa_only'])
    parser.add_argument('--img_h',       type=int,   default=512)
    parser.add_argument('--img_w',       type=int,   default=1024)
    # Backend (MỚI — mặc định 'baseline' giữ đúng hành vi cũ)
    parser.add_argument('--backend',     default='baseline',
                        choices=list(BACKEND_REGISTRY.keys()),
                        help="Backend inference: baseline/fp16/compile/"
                             "compile_fp16/torchscript/onnxruntime/tensorrt/openvino")
    # --- Tham số cho onnxruntime / tensorrt / openvino (đều có default an toàn,
    #     KHÔNG ảnh hưởng CLI cũ nếu không dùng các backend này) ---
    parser.add_argument('--onnx_path', default=None,
                        help='Đường dẫn .onnx (mặc định: checkpoints/gcnet_{h}x{w}.onnx)')
    parser.add_argument('--force_rebuild', action='store_true',
                        help='Ép export lại ONNX / build lại TensorRT engine dù đã có cache')
    parser.add_argument('--trt_engine_path', default=None,
                        help='Đường dẫn .engine (mặc định: tự đặt tên theo onnx_path + precision)')
    parser.add_argument('--trt_precision', choices=['fp32', 'fp16', 'int8'], default='fp16')
    parser.add_argument('--trt_workspace_gb', type=float, default=4)
    parser.add_argument('--calib_dir', default=None,
                        help='Thư mục ảnh calibration cho TensorRT INT8 (QDQ static '
                             'quantization). Không có -> dùng random data, CHỈ demo tốc độ, '
                             'KHÔNG dùng cho production.')
    parser.add_argument('--ov_device', default='AUTO',
                        help='CPU / GPU (iGPU Intel) / NPU / AUTO')
    parser.add_argument('--ov_precision', choices=['FP32', 'FP16'], default='FP16')
    # Benchmark
    parser.add_argument('--benchmark',   action='store_true',
                        help='Chạy speed benchmark (default: True nếu không có --validate)')
    parser.add_argument('--n_warmup',    type=int,   default=50)
    parser.add_argument('--n_repeat',    type=int,   default=3)
    parser.add_argument('--target_sec',  type=float, default=6.0)
    # Validate
    parser.add_argument('--validate',    action='store_true',
                        help='Chạy validation')
    parser.add_argument('--val_txt',     type=str,   default=None)
    parser.add_argument('--batch_size',  type=int,   default=22)
    parser.add_argument('--num_workers', type=int,   default=4)
    parser.add_argument('--no_amp',      action='store_true')
    args = parser.parse_args()

    # Nếu không flag nào được set, default chạy benchmark
    if not args.validate and not args.benchmark:
        args.benchmark = True

    if args.validate and not args.val_txt:
        parser.error("--validate yêu cầu --val_txt")

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    use_amp  = not args.no_amp and device.type == 'cuda'

    print(f"\n{'='*55}")
    print(f"  GCNet v3  —  {'Benchmark' if args.benchmark else ''}"
          f"{'+ ' if args.benchmark and args.validate else ''}"
          f"{'Validate' if args.validate else ''}")
    print(f"{'='*55}")
    print(f"  Device: {gpu_name}")
    print(f"  Input:  {args.img_h}×{args.img_w}  |  AMP: {use_amp}"
          f"  |  Backend: {args.backend}")
    print(f"{'='*55}\n")

    model, recorded = build_model(
        args.model_variant, args.ckpt, device, deploy=True)

    if args.validate:
        validate(model, args.val_txt, args.img_h, args.img_w,
                 args.batch_size, args.num_workers, device, use_amp, recorded)

    if args.benchmark:
        print(f"\nBuilding backend '{args.backend}'...")
        backend_kwargs = dict(
            onnx_path=args.onnx_path,
            force_rebuild=args.force_rebuild,
            trt_engine_path=args.trt_engine_path,
            trt_precision=args.trt_precision,
            trt_workspace_gb=args.trt_workspace_gb,
            calib_dir=args.calib_dir,
            ov_device=args.ov_device,
            ov_precision=args.ov_precision,
        )
        runner = build_runner(args.backend, model, device, args.img_h, args.img_w,
                               **backend_kwargs)

        print(f"Running benchmark...")
        result = benchmark(runner, args.img_h, args.img_w, device,
                           n_warmup=args.n_warmup,
                           n_repeat=args.n_repeat,
                           target_sec=args.target_sec)
        print_benchmark(result, args.model_variant, args.backend,
                         args.img_h, args.img_w, gpu_name)


if __name__ == '__main__':
    main()
