# Chạy khử sương trên Windows (RTX 50 series)

Môi trường: Python 3.12, PyTorch 2.7.1 + CUDA 12.8. Bản PyTorch này
được chọn thay cho 2.5 trong README để hỗ trợ GPU RTX 50 series.

## Cài đặt lại môi trường

Chạy trong PowerShell tại thư mục repo (cần có `uv`):

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-gpu.txt
```

Tải [checkpoints.zip](https://github.com/CunchuanHuang/MDST-Dehaze/releases/download/1.0/checkpoints.zip)
và đặt trọng số để có hai file:

```text
results/checkpoints/example/Sen.pth
results/checkpoints/example/Tde.pth
```

## Chạy ảnh mẫu

```powershell
powershell -ExecutionPolicy Bypass -File .\run-dehaze.ps1
```

Ảnh đầu vào: `datasets/example/test/haze/`.
Ảnh kết quả: `results/dehaze/example/inference/`.

Trong VS Code, chọn interpreter `.venv\Scripts\python.exe`.
Muốn dùng bộ ảnh khác, sửa đường dẫn ảnh và trọng số tương ứng trong
`configs/configs_inferenceH2C.py`. Script gốc cần chạy từ thư mục `scripts`;
`run-dehaze.ps1` tự chuyển thư mục trước khi chạy.
