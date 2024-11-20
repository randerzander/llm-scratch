#llama-cpp
cd ~/projects/llama-cpp-python
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install --upgrade --force-reinstall --no-cache-dir .

# for bertopic
#pip install cudf-cu12 dask-cudf-cu12 --extra-index-url=https://pypi.nvidia.com
#pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com
#pip install cugraph-cu12 --extra-index-url=https://pypi.nvidia.com
