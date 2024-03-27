#llama-cpp
cd ~/projects/llama-cpp-python
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python

# for bertopic
pip install cudf-cu12 dask-cudf-cu12 --extra-index-url=https://pypi.nvidia.com
pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com
pip install cugraph-cu12 --extra-index-url=https://pypi.nvidia.com
