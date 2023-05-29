import optuna


def objective(trial):
    x = trial.suggest_float("x", -1, 1)
    print(trial.number)
    return x**2


study = optuna.create_study()
study.optimize(objective, n_trials=3, n_jobs=-1)
