from __future__ import print_function

import argparse
import json
import logging
import os
import pandas as pd
import pickle as pkl
import numpy as np
import warnings

from sagemaker_containers import entry_point
from sagemaker_xgboost_container.data_utils import get_dmatrix
from sagemaker_xgboost_container import distributed

import xgboost as xgb


def _xgb_train(params, dtrain, evals, num_boost_round, model_dir, is_master, feval, early_stopping_rounds):
    """Run xgb train on arguments given with rabit initialized.

    This is our rabit execution function.

    :param args_dict: Argument dictionary used to run xgb.train().
    :param is_master: True if current node is master host in distributed training,
                        or is running single node training job.
                        Note that rabit_run will include this argument.
    """
    booster = xgb.train(params=params,
                        feval=feval,
                        dtrain=dtrain,
                        evals=evals,
                        num_boost_round=num_boost_round,
                        early_stopping_rounds=early_stopping_rounds)

    if is_master:
        model_location = model_dir + '/xgboost-model'
        pkl.dump(booster, open(model_location, 'wb'))
        logging.info("Stored trained model at {}".format(model_location))

def eval_recall(y, preds, precision_threashold=.5, downsampling_multiplier=1, early_stopping_rounds=10):
    # begin calculation
    labels = y
    tot_bads = sum(labels)
    stacked = np.stack((preds, labels), axis=-1)
    stacked = stacked[stacked[:, 0].argsort()][::-1]  # sorting by highest preds descending
    bad = 0
    total = 0
    esr = 0
    recall = None
    for i in range(len(stacked)):
        bad += stacked[i, 1]
        total += 1
        adj_total = ((total - bad) * downsampling_multiplier) + bad
        precision = round(bad / adj_total, 6)
        if precision <= precision_threashold:
            if esr == 0:
                recall = round(bad / tot_bads, 6)
            esr += 1
        if precision > precision_threashold:
            esr = 0
        if esr >= early_stopping_rounds:
            break
    if recall is None:
        print("\t Your precision is too low. Try setting up to a higher value for recall@precision evaluation "
              "metric. \n You can do that in the trainingConfig.json under "
              "algorithms-settings-precision_threashold. \n Now exiting Eden.")
        exit(1)
    return recall        
        
def eval_recall_xgb(predt, dtrain):
    # from utils.evaluation_utils import eval_recall
    precision_threashold = 0.5
    # downsampling_multiplier should be set to 1 if no downsampling, else divide new by original bad rate
    # eg. downsample: from 0.2% bad rate to 10.0% bad rate, downsampling_multiplier = 0.1/0.0002
    # = 0.025/0.012
    downsampling_multiplier = 1
    # early stopping rounds
    early_stopping_rounds = 20
    y = dtrain.get_label()
    return 'recall@prec', eval_recall(y, predt, precision_threashold, downsampling_multiplier, early_stopping_rounds)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here.
    parser.add_argument('--max_depth', type=int,)
    parser.add_argument('--eta', type=float)
    parser.add_argument('--gamma', type=int)
    parser.add_argument('--min_child_weight', type=int)
    parser.add_argument('--subsample', type=float)
    parser.add_argument('--objective', type=str)
    parser.add_argument('--num_round', type=int)
    parser.add_argument('--early_stopping_rounds', type=int)
    parser.add_argument('--scale_pos_weight', type=int)
    parser.add_argument('--booster', type=str)
        
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output_data_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--sm_hosts', type=str, default=os.environ.get('SM_HOSTS'))
    parser.add_argument('--sm_current_host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    args, _ = parser.parse_known_args()

    # Get SageMaker host information from runtime environment variables
    sm_hosts = json.loads(args.sm_hosts)
    sm_current_host = args.sm_current_host

    dtrain = get_dmatrix(args.train, 'libsvm')
    dval = get_dmatrix(args.validation, 'libsvm')
    watchlist = [(dtrain, 'train'), (dval, 'validation')] if dval is not None else [(dtrain, 'train')]

    train_hp = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'gamma': args.gamma,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'objective': args.objective,
        'scale_pos_weight': args.scale_pos_weight,
        'booster': args.booster,
        }

    xgb_train_args = dict(
        params=train_hp,
        dtrain=dtrain,
        feval=eval_recall_xgb,
        evals=watchlist,
        num_boost_round=args.num_round,
        model_dir=args.model_dir,
        early_stopping_rounds=args.early_stopping_rounds)

    if len(sm_hosts) > 1:
        # Wait until all hosts are able to find each other
        entry_point._wait_hostname_resolution()

        # Execute training function after initializing rabit.
        distributed.rabit_run(
            exec_fun=_xgb_train,
            args=xgb_train_args,
            include_in_training=(dtrain is not None),
            hosts=sm_hosts,
            current_host=sm_current_host,
            update_rabit_args=True
        )
    else:
        # If single node training, call training method directly.
        if dtrain:
            xgb_train_args['is_master'] = True
            _xgb_train(**xgb_train_args)
        else:
            raise ValueError("Training channel must have data to train model.")


def model_fn(model_dir):
    """Deserialize and return fitted model.

    Note that this should have the same name as the serialized model in the _xgb_train method
    """
    model_file = 'xgboost-model'
    booster = pkl.load(open(os.path.join(model_dir, model_file), 'rb'))
    return booster