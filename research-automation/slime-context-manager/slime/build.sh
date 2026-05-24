
conda activate /root/shared-nvme/workspace/envs/slime

shared-nvme/workspace/CM-OPD/research-automation/slime-context-manager/models


hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/shared-nvme/workspace/CM-OPD/research-automation/slime-context-manager/data/dapo-math-17k

 
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY no_proxy NO_PROXY

 
export https_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
export http_proxy="http://u-UE25Z3:tXGJgV92@10.255.128.102:3128"
export no_proxy="127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,*.paracloud.com,*.paratera.com,*.blsc.cn"


Model_name="qwen3-0.6B"
MODELDIR="/root/shared-nvme/workspace/CM-OPD/research-automation/slime-context-manager/models"
source "/root/shared-nvme/workspace/CM-OPD/research-automation/slime-context-manager/slime/scripts/models/${Model_name}.sh"
PYTHONPATH=/root/shared-nvme/workspace/CM-OPD/research-automation/slime-context-manager/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/shared-nvme/workspace/CM-OPD/research-automation/slime-context-manager/models/qwen3-0.6B \
    --save /root/shared-nvme/workspace/CM-OPD/research-automation/slime-context-manager/models/qwen3-0.6B_torch_dist

    
 
# cuda 12.9 (nvcc -V, nvidia-smi)
cd OpenClaw-RL
conda create --name openclaw-rl python=3.12 -y
conda activate openclaw-rl
 
pip install \
  torch==2.9.1+cu129 \
  torchvision==0.24.1+cu129 \
  torchaudio==2.9.1+cu129 \
  --index-url https://download.pytorch.org/whl/cu129
 
pip install -r requirements.txt

# DeepEP
git clone https://github.com/deepseek-ai/DeepEP.git
cd DeepEP
pip install -e . --no-build-isolation
cd ..

pip install -e slime/slime/backends/megatron_utils/kernels/int4_qat --no-build-isolation
 
# apex
git clone https://github.com/NVIDIA/apex.git
cd apex
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .
cd ..

# flash_attn
export MAX_JOBS=8
pip install --no-build-isolation -v flash-attn==2.7.4.post1
 
# flashinfer
pip install "flashinfer-jit-cache==0.6.3" --index-url https://flashinfer.ai/whl/cu129

# megatron-bridge
pip install "megatron-bridge @ git+https://github.com/fzyzcjy/Megatron-Bridge.git@35b4ebfc486fb15dcc0273ceea804c3606be948a" --no-build-isolation

# TransformerEngine
export NVTE_FRAMEWORK=pytorch
pip install --no-build-isolation "transformer_engine[pytorch,core_cu12]==2.10.0"

# apt
apt-get update
apt-get install -y python3-apt