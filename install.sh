rm -rf dev2
uv venv dev2 --python 3.12
source dev2/bin/activate
uv pip install torch torchvision torchaudio
uv pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
uv pip install -r requirements.txt --no-build-isolation

#llama-cpp
#cd ~/projects/llama-cpp-python
#cd ~/projects
#rm -rf llama-cpp-python
#git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git
#cd llama-cpp-python
#CMAKE_ARGS="-DGGML_CUDA=ON" uv pip install --upgrade --force-reinstall --no-cache-dir .
#cd ../llm-scratch

# for bertopic
#pip install cudf-cu12 dask-cudf-cu12 --extra-index-url=https://pypi.nvidia.com
#pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com
#pip install cugraph-cu12 --extra-index-url=https://pypi.nvidia.com
