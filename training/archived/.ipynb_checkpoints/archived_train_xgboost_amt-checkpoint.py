import argparse
import json
import os
import random
import pandas as pd
import glob
import pickle as pkl
import numpy as np
import xgboost

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--min_child_weight", type=int, default=6)
    parser.add_argument("--silent", type=int, default=0)
    
    parser.add_argument('--objective', type=str)
    
#     parser.add_argument("--objective", type=str, default="binary:logistic")
#     parser.add_argument("--eval_metric", type=str, default="auc")
    
    parser.add_argument("--num_round", type=int, default=100)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--early_stopping_rounds", type=int, default=20)

    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))

    args = parser.parse_args()

    return args


# Test Eval Function
def eval_custom_metric(predt, dtrain):
    y = dtrain.get_label()
    predt[predt < -1] = -1 + 1e-6
    elements = np.power(np.log1p(y) - np.log1p(predt), 2)
    custom_metric = float(np.sqrt(np.sum(elements) / len(y)))
    
    return 'multi@obj', custom_metric


def statistical_disparity(model, X, Y, groups):
    """
    :param model: the trained model
    :param X: the input dataset with n observations
    :param Y: binary labels associated to the n observations (1 = positive)
    :param groups: a list of n values binary values defining two different subgroups of the populations
    """
    fY = model.predict(X)
    
    
    sp = [0, 0]
    sp[0] = float(
        len([1 for idx, fy in enumerate(fY) if fy == 1 and groups[idx] == 0])
    ) / len([1 for idx, fy in enumerate(fY) if groups[idx] == 0])
    
    sp[1] = float(
        len([1 for idx, fy in enumerate(fY) if fy == 1 and groups[idx] == 1])
    ) / len([1 for idx, fy in enumerate(fY) if groups[idx] == 1])
    
    return abs(sp[0] - sp[1])

# TODO
def eval_di_metric(predt, dtrain, groups):
    
    #TODO: Get X from dtrain
    
    fY = model.predict(X)
    
    di = computeDisparateImpact(fY, groups)
    
    # TODO: How do we get second metric during training?

    return "multi@obj", multi_obj_metric


def computeDisparateImpact(fY, groups):
    """
    Compute DI metric 
    """
    
    sp = [0, 0]
    
    sp[0] = float(
        len([1 for idx, fy in enumerate(fY) if fy == 1 and groups[idx] == 0])
    ) / len([1 for idx, fy in enumerate(fY) if groups[idx] == 0])
    
    sp[1] = float(
        len([1 for idx, fy in enumerate(fY) if fy == 1 and groups[idx] == 1])
    ) / len([1 for idx, fy in enumerate(fY) if groups[idx] == 1])
    
    di = abs(sp[0] - sp[1])
    
    return di
    

def main():

    args = parse_args()
    train_files_path, validation_files_path = args.train, args.validation

    train_features_path = os.path.join(args.train, "train_features.csv")
    train_labels_path = os.path.join(args.train, "train_labels.csv")

    val_features_path = os.path.join(args.validation, "val_features.csv")
    val_labels_path = os.path.join(args.validation, "val_labels.csv")

    print("Loading training dataframes...")
    df_train_features = pd.read_csv(train_features_path)
    df_train_labels = pd.read_csv(train_labels_path)

    print("Loading validation dataframes...")
    df_val_features = pd.read_csv(val_features_path)
    df_val_labels = pd.read_csv(val_labels_path)

    X = df_train_features.values
    y = df_train_labels.values

    val_X = df_val_features.values
    val_y = df_val_labels.values

    dtrain = xgboost.DMatrix(X, label=y)
    dval = xgboost.DMatrix(val_X, label=val_y)

    watchlist = [(dtrain, "train"), (dval, "validation")]
    
    
    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "gamma": args.gamma,
        "min_child_weight": args.min_child_weight,
        "silent": args.silent,
        "objective": args.objective,
        "subsample": args.subsample,
        "eval_metric": args.eval_metric,
        "early_stopping_rounds": args.early_stopping_rounds,
    }
    
    evals_result = {}
    
    bst = xgboost.train(
        params=params, dtrain=dtrain, feval=eval_custom_metric, evals=watchlist, num_boost_round=args.num_round, evals_result=evals_result
    )
    
    print('Access metric directly from evals_result:')
    print(evals_result['eval'])

    
    #Compute DI metric
    external_di_metric = statistical_disparity(bst, val_X, val_y, val_X[:, -1])
    
    print('external_di_metric:', external_di_metric)
    
    model_dir = os.environ.get("SM_MODEL_DIR")
    
    # TODO: We need to save the model with different names here?
    pkl.dump(bst, open(model_dir + "/model.bin", "wb"))
    
    print("Stored trained model at {}".format(model_dir))
    
    ### from syne-tune example ###
    
    # dsp_foreign_worker = statistical_disparity(
    #     classifier, data_dict["X_test"], data_dict["Y_test"], data_dict["is_foreign"]
    # )
    # y_pred = classifier.predict(data_dict["X_test"])
    # accuracy = accuracy_score(y_pred, data_dict["Y_test"])
    # objective_value = accuracy
    # constraint_value = (
    #     dsp_foreign_worker - args.fairness_threshold
    # )  # If DSP < fairness threshold,
    # # then we consider the model fair
    # report(objective=objective_value, my_constraint_metric=constraint_value)


if __name__ == "__main__":
    main()
