import json, torch, subprocess
info={
 "torch": torch.__version__,
 "cuda_available": torch.cuda.is_available(),
 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(info, ensure_ascii=False, indent=2))