import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from patchTST import Model as PatchTST

# Initialize and fetch data from MetaTrader 5
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

# Fetch data
data = fetch_mt5_data('EURUSD', 'H1', 80000)

# Prepare forecasting data using sliding window
def prepare_forecasting_data(data, seq_length, pred_length):
    X, y = [], []
    for i in range(len(data) - seq_length - pred_length):
        X.append(data.iloc[i:(i + seq_length)].values)
        y.append(data.iloc[(i + seq_length):(i + seq_length + pred_length)].values)
    return np.array(X), np.array(y)

seq_length = 168  # 1 week of hourly data
pred_length = 24  # Predict next 24 hours

X, y = prepare_forecasting_data(data, seq_length, pred_length)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the configuration class
class Config:
    def __init__(self):
        self.enc_in = 4  # Adjusted for 4 columns (open, high, low, close)
        self.seq_len = seq_length
        self.pred_len = pred_length
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

configs = Config()

# Initialize the PatchTST model
model = PatchTST(
    configs=configs,
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
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_X)
        outputs = outputs[:, -pred_length:, :4]

        loss = loss_fn(outputs, batch_y)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.10f}")

torch.save(model.state_dict(), 'patchtst_model.pth')

# Prepare a dummy input for ONNX export
dummy_input = torch.randn(1, seq_length, 4).to(device)

# Export to ONNX
torch.onnx.export(model, dummy_input, "patchtst_model.onnx",
                  opset_version=13,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'output': {0: 'batch_size'}})

print("Model trained and saved in PyTorch and ONNX formats.")


"""Use code for troubleshooting


dummy_input = torch.randn(1, seq_length, 4).to(device)

# Convert the PyTorch model to a TorchScript model
scripted_model = torch.jit.trace(model, dummy_input)

# Save the TorchScript model to a file
torch.jit.save(scripted_model, "patchtst_model.pt")


# Export the TorchScript model to ONNX
torch.onnx.export(scripted_model, dummy_input, "patchtst_model.onnx",
                  opset_version=13,
                  input_names=['input'],
                  output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size', 1: 'sequence_length'},
                                'output': {0: 'batch_size', 1: 'sequence_length'}})

print("Model trained and saved in PyTorch and ONNX formats.")"""