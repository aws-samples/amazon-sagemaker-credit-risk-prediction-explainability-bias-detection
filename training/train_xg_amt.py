import argparse
import json
import os
import random
import pandas as pd
import glob
import pickle as pkl
import numpy as np
import xgboost
import statistics
import boto3
from dmatrix2np import dmatrix_to_numpy
from sklearn.metrics import roc_auc_score

s3 = boto3.client('s3')

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--min_child_weight", type=int, default=6)
    parser.add_argument("--silent", type=int, default=0)
    
    parser.add_argument("--objective", type=str)
    parser.add_argument("--eval_metric", type=str)   # TODO: Double check 
    
    parser.add_argument("--num_round", type=int, default=100)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--early_stopping_rounds", type=int, default=20)

    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    
    args = parser.parse_args()

    return args

def write_to_s3(di, auc_score):
    args = parse_args()
    # Save a temp text file to output_data_dir
    f = open(os.path.join(args.output_data_dir, 'raw_ouptut.txt'), 'a+')
    f.write(str(di) + "," + str(auc_score) + "\n")
    f.close()
            
def eval_auc_score(predt, dtrain):
    fY = [1 if p > 0.5 else 0 for p in predt]
    y = dtrain.get_label()
    auc_score = roc_auc_score(y, fY)
    
    return auc_score

# Combined metrics for SD and AUC
def eval_combined_metric(predt, dtrain):
    auc_score = eval_auc_score(predt, dtrain)
    sd = eval_statistical_disparity(predt, dtrain)
    combined_metric = ((4*auc_score)+(1-sd))/5
    print("Statistical Disparity, AUC Score, Combined Metric: ", sd, auc_score, combined_metric)
    write_to_s3(round(sd,4), round(auc_score,4))
    
    return "auc", combined_metric

def eval_statistical_disparity(predt, dtrain):
    """
    Eval SD metric 
    fY - prediction [ credit risk - 0: bad 1: good]
    groups - Foreign worker [ 1 yes, 2 no]
    """
    
    dtrain_np = dmatrix_to_numpy(dtrain)

    # Foreign worker 
    groups = [item for sublist in dtrain_np[:, -1:] for item in sublist]
    
    fY = [1 if p > 0.5 else 0 for p in predt]
    sp = [0, 0]
    sp[0] = float(
        len([1 for idx, fy in enumerate(fY) if fy == 1 and groups[idx] == 1.0])
    ) / len([1 for idx, fy in enumerate(fY) if groups[idx] == 1.0])
    
    sp[1] = float(
        len([1 for idx, fy in enumerate(fY) if fy == 1 and groups[idx] == 2.0])
    ) / len([1 for idx, fy in enumerate(fY) if groups[idx] == 2.0])
    
    sd = abs(sp[0] - sp[1])
    
    return sd
    

def main():

    args = parse_args()
    
    train_files_path, validation_files_path = args.train, args.validation

    train_features_path = os.path.join(args.train, "train_features.csv")
    train_labels_path = os.path.join(args.train, "train_labels.csv")

    val_features_path = os.path.join(args.validation, "val_features.csv")
    val_labels_path = os.path.join(args.validation, "val_labels.csv")

    #Loading training dataframes
    df_train_features = pd.read_csv(train_features_path)
    df_train_labels = pd.read_csv(train_labels_path)

    #Loading validation dataframes
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
        "early_stopping_rounds": args.early_stopping_rounds
    }
    
    evals_result = {}
    
    bst = xgboost.train(
        params=params, dtrain=dtrain, feval=eval_combined_metric, evals=watchlist, num_boost_round=args.num_round, evals_result=evals_result
    )
    
    # Write the results to S3 bucket (optional) - only to plot 'Pareto frontier'
    import uuid
    import re
    raw_df = pd.read_csv(args.output_data_dir + 'raw_ouptut.txt',  sep=',', header = None)
    raw_file_name = 'raw_ouptut.txt'
    sorted_file_name = "sorted_output.txt"
    df = pd.read_csv(args.output_data_dir + raw_file_name, sep=",", header=None)
    # get last row
    df = df.iloc[-1:]
    df.to_csv(args.output_data_dir + sorted_file_name, sep=',', index=False, header=False)
    
    random_id = str(uuid.uuid4().hex)[:8]                                            
    out_file_name = '{}-output.txt'.format(random_id)               

    s3_module_dir = os.environ.get("SM_MODULE_DIR")
    tmp_s3_bucket_path = s3_module_dir.replace("s3://","")
    s3_bucket_name = re.split('[/]',tmp_s3_bucket_path)[0]
    s3_bucket_prefix = "sagemaker/sagemaker-amt-credit-risk-model/data/output/tmp/"
    
    with open(args.output_data_dir + sorted_file_name, "rb") as data:
        s3.upload_fileobj(data, s3_bucket_name, s3_bucket_prefix + out_file_name)
    
    # Write model
    model_dir = os.environ.get("SM_MODEL_DIR")
    pkl.dump(bst, open(model_dir + "/model.bin", "wb"))
    print("Stored trained model at {}".format(model_dir))

if __name__ == "__main__":
    main()
    
    
    
