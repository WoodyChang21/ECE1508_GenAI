import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from patchTST import Model as PatchTST

def fetch_mt5_data(symbol, timeframe, bars):
    if not mt5.initialize():
        print("MT5 initialization failed")
        return None

    timeframe_dict = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'H1': mt5.TIMEFRAME_H1,
        'D1': mt5.TIMEFRAME_D1
    }

    rates = mt5.copy_rates_from_pos(symbol, timeframe_dict[timeframe], 0, bars)
    mt5.shutdown()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df[['open', 'high', 'low', 'close']]

def prepare_input_data(data, seq_length):
    X = []
    X.append(data.iloc[-seq_length:].values)
    return np.array(X)

class Config:
    def __init__(self):
        self.enc_in = 4
        self.seq_len = 168  # 1 week of hourly data
        self.pred_len = 24  # Predict next 24 hours
        self.e_layers = 3
        self.n_heads = 4
        self.d_model = 64
        self.d_ff = 256
        self.dropout = 0.1
        self.fc_dropout = 0.1
        self.head_dropout = 0.1
        self.individual = False
        self.patch_len = 24
        self.stride = 24
        self.padding_patch = True
        self.revin = True
        self.affine = False
        self.subtract_last = False
        self.decomposition = True
        self.kernel_size = 25

def load_model(model_path, config):
    model = PatchTST(
        configs=config,
        max_seq_len=1024,
        d_k=None,
        d_v=None,
        norm='BatchNorm',
        attn_dropout=0.1,
        act="gelu",
        key_padding_mask='auto',
        padding_var=None,
        attn_mask=None,
        res_attention=True,
        pre_norm=False,
        store_attn=False,
        pe='zeros',
        learn_pe=True,
        pretrain_head=False,
        head_type='flatten',
        verbose=False
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, input_data, device):
    with torch.no_grad():
        input_data = torch.tensor(input_data, dtype=torch.float32).to(device)
        output = model(input_data)
    return output.cpu().numpy()

def make_prediction(model_path, config, latest_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_model(model_path, config).to(device)
    input_data = prepare_input_data(latest_data, config.seq_len)
    predictions = predict(model, input_data, device)
    
    return predictions

if __name__ == "__main__":
    config = Config()

    # Fetch the latest week of data
    historical_data = fetch_mt5_data('EURUSD', 'H1', 168)

    model_path = 'patchtst_model.pth'
    predictions = make_prediction(model_path, config, historical_data)

    # Ensure predictions have the correct shape
    if predictions.shape[2] != 4:
        predictions = predictions[:, :, :4]  # Adjust based on actual number of columns required

    # Check the shape of predictions
    print("Shape of predictions:", predictions.shape)

    # Create a DataFrame for predictions
    pred_index = pd.date_range(start=historical_data.index[-1] + pd.Timedelta(hours=1), periods=24, freq='H')
    pred_df = pd.DataFrame(predictions[0], columns=['open', 'high', 'low', 'close'], index=pred_index)

    # Combine historical data and predictions
    combined_df = pd.concat([historical_data, pred_df])

    # Create the plot
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=('EURUSD OHLC'))

    # Add historical candlestick
    fig.add_trace(go.Candlestick(x=historical_data.index,
                                 open=historical_data['open'],
                                 high=historical_data['high'],
                                 low=historical_data['low'],
                                 close=historical_data['close'],
                                 name='Historical'))

    # Add predicted candlestick
    fig.add_trace(go.Candlestick(x=pred_df.index,
                                 open=pred_df['open'],
                                 high=pred_df['high'],
                                 low=pred_df['low'],
                                 close=pred_df['close'],
                                 name='Predicted'))

    # Add a vertical line to separate historical data from predictions
    fig.add_vline(x=historical_data.index[-1], line_dash="dash", line_color="gray")

    # Update layout
    fig.update_layout(title='EURUSD OHLC Chart with Predictions',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False)

    # Show the plot
    fig.show()

    # Print predictions (optional, you can remove this if you don't need it)
    print("Predicted prices for the next 24 hours:", predictions)
