## Installation
```bash
conda create -n fable python=3.9.7
pip install -r requirements.txt
```

## Data Preparation
The relevant datasets need to be downloaded from [https://huggingface.co/datasets/wpwpyo/UnFine](https://huggingface.co/datasets/wpwpyo/UnFine) to the local `./data` folder.

## Edit
### 1. Edit LLAMA3-8B Model on UnFine-UnKE using FABLE
```bash
run_fable.sh
```

Results from each run are stored at `results/<ds_name>/<model_name>/<method_name>/<args_name>/post/` in a specific format:
```bash
results/
|__ unke/
    |__ Llama3-8B-Instruct/
        |__ FABLE/
            |__ <args_name>/
                |__ post/
                    |__ 0.json
                    |__ 1.json
                    |__ ...
                    |__ 999.json
```

If you want to run the other baselines, you can ferer to run_baseline.sh

#### 2. Summarize the results
```bash
bash compute_score.sh results/<ds_name>/<model_name>/<method_name>/<args_name>/post
```


## Acknowledgment
Our code is based on  [``EasyEdit``](https://github.com/zjunlp/EasyEdit), and [``UnKE``](https://github.com/TrustedLLM/UnKE).
