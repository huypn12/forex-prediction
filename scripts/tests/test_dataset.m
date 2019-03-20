cfg = config();

eurusd = fx_dataset(cfg.dataset_sample, "");

std_eurusd = fx_standardize(eurusd);
trainset = fx_trainset(std_eurusd, 100);