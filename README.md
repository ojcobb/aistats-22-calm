# Sequential Multivariate Change Detection with Calibrated and Memoryless False Detection Rates (AISTATS 2022)

To run the experiments in Section 5.1 first install poetry if you do not already have it:
```
pip install poetry
```
Then instantiate an environment and install the required dependencies
```
poetry shell
poetry install
```
This doesn't install `pytorch`. Do this following the instructions at https://pytorch.org/.

To run experiments then run
```
cd experiments/toy_examples
python run.py
```
This obtains ARTs and ADDs for desired ERTs of 128, 256, 512, 1024. In the paper we perform additional experiments for `batchmmdsim` and `lsddincsim` with ERTs set equal to the ARTs obtained by `bstat` and `lsddinc` respectively. This can be performed by updating the `ert` parameters in `gen_config.yaml` files.

---
Running the experiments in Section 5.2 requires a couple of additional dependencies which are also better installed manually. First run
```
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install wilds==1.1.0
```
being careful to ensure `${TORCH}` and `${CUDA}` match your torch installation and CUDA version. 

Then navigate into `experiments/camelyon` and run `run.py`. It will take a while to download the data the firts time this is run.