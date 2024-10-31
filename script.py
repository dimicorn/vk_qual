import torch
import torch.nn as nn
import pandas as pd
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesResampler
from tqdm import tqdm
from warnings import filterwarnings
import numpy as np


filterwarnings('ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=1000, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size)
        
        self.linear = nn.Linear(hidden_size, output_size)
        
        self.hidden = (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))

    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(seq.view(len(seq), 1, -1), self.hidden)
        pred = self.linear(lstm_out.view(len(seq), -1))
        return pred[-1]

model = LSTM()
model.to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

test_df = pd.read_parquet('test.parquet')
idx = []
for i, value_lst in enumerate(test_df['values']):
    if np.isnan(value_lst).any() == True:
        idx.append(i)

test_df = test_df.drop(idx)
test_df = test_df.reset_index(drop=True)

X = to_time_series_dataset(test_df['values'])
resampled_X = TimeSeriesResampler(sz=36).fit_transform(X)
test_data = torch.tensor(resampled_X.squeeze(), dtype=torch.float32)
preds_val = []

with torch.no_grad():
    for seq in tqdm(test_data):
        seq = seq.to(device)
        output = model(seq)
        preds_val.append(float(output))

submission = pd.DataFrame({'id': test_df['id'], 'score': preds_val})
submission.to_csv('submission.csv', index=False)