# Secure N-Agent Ad-hoc Teamwork

## WHAT ARE DBBC AND RAPO? :open_mouth:

This repository represents an updated iteration of the original framework, now implementing **DBBC (Distributed Bayesian Belief Consensus)** and **RAPO (Risk-Aware Policy Optimization)** architectures. The legacy BAE files have been largely removed or integrated. The codebase now focuses on handling adversarial agent detection in an ad-hoc setting through distributed evidence fusion (DBBC) and decentralized risk-aware reinforcement learning execution (RAPO).

## SUMMARY

In this README you can find:
- [WHAT ARE DBBC AND RAPO?](#what-are-dbbc-and-rapo-open_mouth)
- [GET STARTED](#get-started)
  - [1. Dependencies](#1-dependencies-pencil)
  - [2. Usage Workflow](#2-usage-workflow-muscle)
- [REFERENCES](#references)

## GET STARTED

### 1. Dependencies :pencil:

<b>- About this repository</b>

This repository relies on PyTorch as well as several standard machine learning packages. Ensure your virtual environment is properly configured.
Install the dependencies using:

```bash
pip install -r requirements.txt
```

### 2. Usage Workflow :muscle:

The current project workflow incorporates data collection, DBBC fusion models, and RAPO distributed execution.

<b>1. Data Collection</b>
Run simulations to collect observation/action histories for training DBBC models. The output logs will be stored as CSV files inside the `results/` folder.
```bash
python collect_data.py
```

<b>2. Train Modalities (DBBC, RAPO, and Baselines)</b>
You can use the various training scripts included in the repository depending on which framework layer you wish to build:
- **`train_dbbc.py`**: Trains the Distributed Bayesian Belief Consensus module (`dbbc_pretrained.pth`) to perform fusion for adversary detection.
- **`train_rapo.py`**: Trains the Risk-Aware Policy Optimization PPO agent (`rapo_pretrained.pth`) on top of the DBBC structures.
- **`train_baselines.py`**: Runs training loops for various baseline models.

<b>3. Run & Evaluate the Methods</b>
To evaluate DBBC predictions using simulated CSV records:
```bash
python evaluate_dbbc.py
```
To run RAPO in our testing scenarios taking into consideration fully distributed inference natively:
```bash
python run_rapo.py
```

## REFERENCES

<a name="alves2024amongus">[1]</a> Matheus Aparecido do Carmo Alves, Amokh Varma, Yehia Elkhatib, and Leandro Soriano Marcolino. 2024. <b>It Is Among Us: Identifying Adversaries in Ad-hoc Domains Using Q-valued Bayesian Estimations</b>. In Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems (AAMAS '24). Auckland, New Zealand.

<a name="alves2022adleapmas">[2]</a> Matheus Aparecido do Carmo Alves, Amokh Varma, Yehia Elkhatib, and Leandro Soriano Marcolino. 2022. <b>AdLeap-MAS: An Open-source Multi-Agent Simulator for Ad-hoc Reasoning</b>. In Proceedings of the 21st International Conference on Autonomous Agents and Multiagent Systems (AAMAS '22). International Foundation for Autonomous Agents and Multiagent Systems, Richland, SC, 1893–1895.
