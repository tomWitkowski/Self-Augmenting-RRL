macd_cfg = {'EURUSD':{'short': 54, 'long': 237},
 'GBPUSD':{'short': 58, 'long': 139},
 'USDCHF':{'short': 40, 'long': 131},
 'USDJPY':{'short': 12, 'long': 97}
 }

macd_cfg = {
'EURUSD':{'short': 58, 'long': 222, 'ma_train': -0.004199},
'GBPUSD':{'short': 57, 'long': 201, 'ma_train': -0.004166},
'USDCHF':{'short': 44, 'long': 225, 'ma_train': -0.002749},
'USDJPY':{'short': 49, 'long': 180, 'ma_train': -0.005136}}

data_list = ['GBPUSD','EURUSD', 'USDJPY','USDCHF']

import copy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings; warnings.filterwarnings('ignore')
from math import log, factorial
from tqdm import tqdm
import matplotlib as mpl
import os
from tqdm import tqdm
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import seaborn as sns
from scipy.optimize import minimize
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


from src.rml_model import Encoder, Decoder, Agent

model_config_e = [   
    {'type': 'dense', 'units': 5, 'activation': 'softplus', 'dropout': 0.3},
]
model_config_d = [   
    {'type': 'dense', 'units': 5, 'activation': 'softplus', 'dropout': 0.3},
]

def add_ma(data: pd.DataFrame, period_short: int, period_long: int, d_prev: int, ema:bool=True):
    if ema:
        data['ma_short'] = data['bid'].ewm(span=period_short).mean()
        data['ma_long'] = data['bid'].ewm(span=period_long).mean()
    else:
        data['ma_short'] = data['bid'].rolling(window=period_short).mean()
        data['ma_long'] = data['bid'].rolling(window=period_long).mean()
    data['diff'] = data['ma_short'] - data['ma_long']
    for i in range(1, d_prev+1):
        data[f'diff_{i}'] = data['diff'].shift(i)
    return data.dropna(inplace=False)

def ma(x, revert, threshold):
    revert = np.sign(revert)
    return revert*(1 if x > threshold else (-1 if x < -threshold else 0))

# Define the objective function to minimize (negative Sharpe Ratio)
def objective(params, X_train, BA_train, agent):
    threshold, revert = params
    dec_ma = X_train['diff'].map(lambda x: ma(x, threshold=threshold, revert=revert))
    sr_train = agent.utility_function([BA_train.values, dec_ma.values]).numpy()
    return sr_train


def optimize_ma(X_train, BA_train, agent):
    best_objective_value = float('-inf')
    best_params = None

    # Define the range of values to test
    thresholds = np.linspace(0.0001, 1.5, 20)  # 1500 points between 0.0001 and 1.5
    reverts = [-1, 1]

    # Iterate over all combinations of parameters
    for threshold in thresholds:
        for revert in reverts:
            params = (threshold, revert)
            objective_value = objective(params, X_train, BA_train, agent)

            # Check if this is the best set of parameters found so far
            if objective_value > best_objective_value:
                best_objective_value = objective_value
                best_params = params

    return best_params


PREV_HOURS=12
n_runs = 20
epochs = 40
lr = 0.0002
os.environ['train_log_name'] = '20'

results = []
def main(name):
    print(name)
    path = f'data/processed/{name}_15min.csv' 
    data = pd.read_csv(path)
    s,l = macd_cfg[name]['short'], macd_cfg[name]['long']

    df = add_ma(data.copy(), s, l, PREV_HOURS)
    X = df[[col for col in df.columns if 'diff' in col]]
    BA = df[['bid', 'ask']]

    data['time'] = pd.to_datetime(data['time'])
    df['time'] = data.loc[df.index, 'time']

    n = int(len(df) * 0.8)
    idx_train = df.index[:n]
    idx_test = df.index[n:]

    X_train, X_test = X.loc[idx_train], X.loc[idx_test]
    BA_train, BA_test = BA.loc[idx_train], BA.loc[idx_test]

    encoder = Encoder(2, {}); decoder = Decoder(encoder.output.shape[1], {})
    agent = dummy_agent = Agent(encoder, decoder)

    threshold, revert = optimize_ma(X_train, BA_train, agent=agent)
    
    dec_ma_train = X_train['diff'].map(lambda x: ma(x, threshold=threshold, revert=revert))
    sr_ma_train = agent.utility_function([BA_train.values, dec_ma_train.values]).numpy()

    dec_ma_test = X_test['diff'].map(lambda x: ma(x, threshold=threshold, revert=revert))
    sr_ma_test = agent.utility_function([BA_test.values, dec_ma_test.values]).numpy()

    X['ma'] = X['diff'].map(lambda x: ma(x, threshold=threshold, revert=revert))

    multiplier = 1 if name == 'USDJPY' else 100 # 100
    X *= multiplier
    X['ma'] /= multiplier

    X_train, X_test = X.loc[idx_train], X.loc[idx_test]

    sr_test=0
    rml_train_srs = []
    rml_test_srs  = []
    finished = 0
    while finished < n_runs:
        broken=False
        encoder = Encoder(X.shape[1], model_config_e)
        decoder = Decoder(encoder.output.shape[1], model_config_d)
        agent = Agent(encoder, decoder)
        agent.set_lr(lr)

        if n_runs<=2:
            finished += 1
            continue

        srs = [ ]
        srs_train = []
        var = []
        var_train = []

        for i in tqdm(range(int(epochs/20))):
            agent.fit(X_train, BA_train, epochs=20, verbose=False)
            _, decs, sr, _, dec_pred = agent.test_iteration(X_train, BA_train) 
            var_train.append((np.array(decs)**2).mean()**0.5)
            srs_train.append(sr.numpy())
            _, decs, sr, _, dec_pred = agent.test_iteration(X_test, BA_test) 

            if len(srs) == 0 or sr.numpy() > max(srs):
                best_encoder_we = encoder.get_weights()
                best_decoder_we = decoder.get_weights()

            var.append((np.array(decs)**2).mean()**0.5)
            srs.append(sr.numpy())

            if tf.math.is_nan(sr):
                broken=True
                break

        if broken:
            continue
        finished += 1

        if not np.isnan(sr_test): 
            rml_test_srs.append(sr_test)

        _, decs, sr_train, _, _ = agent.test_iteration(X_train, BA_train)
        sr_train = sr_train.numpy()

        if not np.isnan(sr_train):
            rml_train_srs.append(sr_train)
            
        _, decs, sr_test, _, _ = agent.test_iteration(X_test, BA_test)
        sr_test = sr_test.numpy()

        best_encoder = Encoder(X.shape[1], model_config_e)
        best_decoder = Decoder(best_encoder.output.shape[1], model_config_d)
        best_encoder.set_weights(best_encoder_we)
        best_decoder.set_weights(best_decoder_we)
        best_agent = Agent(best_encoder, best_decoder)

        _, decs_test_best, sr_test_best, _, _ = best_agent.test_iteration(X_test, BA_test)
        sr_test_best = sr_test_best.numpy()

        base_catalogue = f"results_{name}/"
        if not os.path.exists(base_catalogue):
            os.makedirs(base_catalogue)
        catalogue_folder = os.path.join(base_catalogue, f"{s}_{l}_{round(100*sr_test,5)}_{finished}")
        if not os.path.exists(catalogue_folder):
            os.makedirs(catalogue_folder)

        np.save(os.path.join(catalogue_folder, "decoder_weights.npy"), decoder.get_weights())
        np.save(os.path.join(catalogue_folder, "encode_weights.npy"), encoder.get_weights())

        df = pd.DataFrame({
            "var": var,
            "var_train": var_train,
            "srs": srs,
            "srs_train": srs_train,
        })
        df.to_csv(os.path.join(catalogue_folder, 'df.csv'))

        df_save = pd.DataFrame({
            'MA': dec_ma_test.values,
            'RRL': decs,
            'RRL_best': decs_test_best,
            'bid': BA_test.bid.values,
            'ask': BA_test.ask.values,
        })
        df_save.to_csv(os.path.join(catalogue_folder, f'positions_test.csv'), index=False)


if __name__ == '__main__':
    multiprocess = True

    if not multiprocess:
        for name in data_list:
            main(name)
    else:
        import multiprocessing

        with multiprocessing.Pool(processes=4) as pool:
            pool.map(main, data_list)
