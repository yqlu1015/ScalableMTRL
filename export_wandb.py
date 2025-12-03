import wandb

api = wandb.Api()
# run = api.run("/wides/MT50/runs/1764415697_mt50_smoe_e6_se0_k4_lb_seed1_1")
run = api.run("/wides/MT50/runs/1764293537_mt50_pcgrad_seed1_1")

history_df = run.history()

history_df.to_csv(f"wandb_logs/{run.name}_data.csv", index=False)
print(f"Data saved for run: {run.name}")