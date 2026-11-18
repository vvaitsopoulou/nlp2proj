#!/bin/bash
# One-shot setup for a fresh vast.ai box (1x RTX 5090). Run from the repo root
# (/workspace/SNLI-stage-2-nlp) AFTER your code is on the box.
#
# IMPORTANT: this does NOT install or touch torch — it keeps the image's own
# PyTorch build, which is the one compatible with the 5090 (Blackwell / sm_120).
# If step 1 fails, your image's torch is too old; pick a newer PyTorch template.
set -e

echo "== 1. GPU / torch sanity (self-healing for Blackwell / RTX 5090) =="
# Try the image's own torch first. If it can't run a kernel on this GPU (the
# classic sm_120 'no kernel image' on a 5090 with an older torch), install the
# CUDA 12.8 build, which has Blackwell kernels. Only runs if the first test fails.
if ! python -c "import torch; assert torch.cuda.is_available(); (torch.randn(64,64,device='cuda')@torch.randn(64,64,device='cuda')).sum().item()" 2>/dev/null; then
  echo "!! this image's torch can't drive the GPU — installing the CUDA 12.8 build (Blackwell)..."
  pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch \
    || pip install --upgrade --break-system-packages --index-url https://download.pytorch.org/whl/cu128 torch
fi
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
python -c "import torch; x=torch.randn(128,128,device='cuda'); print('gpu matmul ok, sum =', float((x@x).sum()))"

echo "== 2. python deps (everything EXCEPT torch) =="
pip install -q numpy scikit-learn scipy nltk datasets pandas huggingface_hub \
  || pip install -q --break-system-packages numpy scikit-learn scipy nltk datasets pandas huggingface_hub

echo "== 3. NLTK data =="
python - <<'PY'
import nltk
for p in ("punkt","punkt_tab","stopwords","wordnet","omw-1.4"):
    try: nltk.download(p, quiet=True)
    except Exception as e: print("skip", p, e)
print("nltk ok")
PY

echo "== 4. GloVe 6B (50/100/300d) -> ./glove =="
mkdir -p glove
if [ ! -f glove/glove.6B.300d.txt ]; then
  python - <<'PY'
import zipfile
try:
    from huggingface_hub import hf_hub_download
    z = hf_hub_download(repo_id="stanfordnlp/glove", filename="glove.6B.zip", repo_type="model")
except Exception as e:
    print("HF mirror failed, falling back to Stanford:", e)
    import urllib.request
    z = "glove.6B.zip"; urllib.request.urlretrieve("https://nlp.stanford.edu/data/glove.6B.zip", z)
with zipfile.ZipFile(z) as f:
    for n in ("glove.6B.50d.txt","glove.6B.100d.txt","glove.6B.300d.txt"):
        f.extract(n, "glove")
print("glove extracted")
PY
else
  echo "glove already present"
fi
ls -la glove

echo ""
echo "== setup done. Test the pipeline end-to-end with the variance study: =="
echo "   python -u run_sweep.py --glove_dir ./glove --results ./results_p2 --variance"