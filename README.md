
## Steps to run

**Experiment 1**
1. Create environment using `requirements.txt`
2. Change into folder `Experiment_1`
3. Run each of the files named `Experiment1-setting1.py` through `Experiment1-setting5.py`.
    - Enable `mlflow` usage in each file by setting `MLFLOW_ENABLED` to `True`. Plotting uses `mlflow` logs to generate the plots.
4. Run `plot.py`


**Experiment 2**
1. Create environment using `requirements.txt`
2. Change into folder `Experiment_2`
3. Run file `Experiment2.py`.
    - Enable `mlflow` usage by setting `MLFLOW_ENABLED` to `True`. Plotting uses `mlflow` logs to generate the plots.
4. Run `plot.py`


**Experiment 3**
1. Create environment using `requirements.txt`
2. Change into folder `Experiment_3`
3. Run file `Experiment3.py`.
    - Enable `mlflow` usage by setting `MLFLOW_ENABLED` to `True`. Plotting uses `mlflow` logs to generate the plots.
4. Run `plot.py`

**Experiment 4**
1. Create environment using `requirements.txt`
2. Change into folder `Experiment_2`
3. Run different variations of the experiment by changing `num_value_c0` in `Experiment2.py`. Log each variation under a different `mlflow` experiment name.
4. Change into folder `Experiment_4`
5. Set `EXPERIMENT_NAMES` in `plot.py` to the list of experiments names used above for the different variations.
6. Run `plot.py`

**Experiment 5**
1. Create environment using `requirements.txt`
2. Change into folder `Experiment_2`
3. Run different variations of the experiment by changing the `WARM_START_SAMPLES` in `Experiment2.py`. Log each variation under a different `mlflow` experiment name.
4. Change into folder `Experiment_5`
5. Set `EXPERIMENT_NAMES` in `plot.py` to the list of experiments names used above for the different variations.
6. Run `plot.py`

**Experiment 6**
1. Create environment using `requirements.txt`
2. Run Experiment 2 as per above. 
3. Then change to folder `Experiment_6` and run `Experiment6-variation.py`. Ensure to use a different `mlflow` experiment name for this.
4. Set `EXPERIMENT_NAMES` in `plot.py` to the both experiments names used above.
5. Run `plot.py`

**Experiment 7a**
1. Create environment using `requirements.txt`
2. Run Experiment 2 as per above. 
3. Then change to folder `Experiment_7` and run `Experiment7a-variation.py`. Ensure to use a different `mlflow` experiment name for this.
4. Set `EXPERIMENT_NAMES` in `plot.py` to the both experiments names used above.
5. Run `plot.py`


## Details of parameterizations
For Experiments 1-3, within each experiment folder,
* The file `env_setting.py` (or `env_setting_*.py`) specifies the domain of each variable
* The function `sample_true_prob` (in the main experiment file named `Experiment*.py`) specifies the parameterization of the causal model

For Experiments 4-7a, the paremeterizations are the same as Experiment 2.

## Details of run parameters
In all experiments, the file named `run_params.py` contains these details. 