# importing libraries
import torch.nn as nn
import numpy as np
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt

# setting the device to run computations on
device = "mps"

# initializing the LSTM hyperparameters
input_size = 1
hidden_size = 150
output_size = 1
batch_size = 10
seq_len = 100
num_layers = 2
dropout = 0.2
learning_rate = 0.01
training_epochs = 1000
eval_interval = 100

# setting the random seed for reproducibility
torch.manual_seed(462)

# sourcing the dataset and preprocessing it
current_dir = os.getcwd()
df_dir = os.path.join(current_dir, "beer.csv")

dataset = pd.read_csv(df_dir)

def preprocess(dataset):
    df = dataset.copy()
    # split_cols = df['Month'].str.split('-', expand=True)
    # df['Year'] = split_cols[0].astype(int)
    # df['Month'] = split_cols[1].astype(int)

    df = df[['Monthly beer production']]
    
    n = int(len(df) * 0.75)
    data = {
        'train': df[:n],
        'test': df[n:]
    }

    return data

data = preprocess(dataset)

# LSTM trains on batches of sequential data, in this case a time series
def get_batch(type = "train", batch_size = batch_size):
    
    batch_df = data[type]
    batch_df_n = len(batch_df)

    x_batch = []
    y_batch = []

    for _ in range(batch_size):

        index = torch.randint(low = 0, high = batch_df_n - seq_len , size = (1,)).item()
        
        x = batch_df.iloc[index:index + seq_len].values # shape (seq_len, input_size)
        y = batch_df.iloc[index + 1:index + seq_len + 1].values

        x = torch.tensor(x, dtype=torch.float32, device=device)
        y = torch.tensor(y, dtype=torch.float32, device=device)

        x_batch.append(x)
        y_batch.append(y)

    # Stack tensors along a new dimension (batch dimension)
    x_batch = torch.stack(x_batch)  # shape (batch_size, seq_len, input_size)
    y_batch = torch.stack(y_batch)  # shape (batch_size, seq_len, input_size)


    return x_batch, y_batch

# defining the loss function (adding an irreducible epsilon term to prevent gradient explosion during 0 RMSE)
class RMSEloss(nn.Module):
    def __init__(self, eps = 1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)
    

# defining the loss calculating function
@torch.no_grad()
def calculate_loss(type = "train"):

    model.eval()

    x, y = get_batch(type)
    lossfn = RMSEloss()

    y_pred = model.evaluate(x)

    loss = lossfn(y_pred, y)
    
    model.train()
    
    return loss


# defining the LSTM 
class Forget_Gate(nn.Module):

    def __init__(self,input_size, hidden_size):
        super().__init__()
        self.x_input = nn.Linear(input_size, hidden_size)
        self.stm_input = nn.Linear(hidden_size, hidden_size)
        self.final = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, x, stm):
        x = self.x_input(x)
        stm = self.stm_input(stm)
        output = x + stm
        output = self.final(output)

        return output
    

class Update_Gate(nn.Module):

    def __init__(self,input_size, hidden_size):
        super().__init__()
        self.x_input_retain_block = nn.Linear(input_size, hidden_size)
        self.stm_input_retain_block = nn.Linear(hidden_size, hidden_size)
        self.final_retain_block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

        self.x_input_pred_block = nn.Linear(input_size, hidden_size)
        self.stm_input_pred_block = nn.Linear(hidden_size, hidden_size)
        self.final_pred_block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

    def forward(self, x, stm):
        
        x_retain = self.x_input_retain_block(x)
        stm_retain = self.stm_input_retain_block(stm)
        x_retain = x_retain + stm_retain
        retain_wei = self.final_retain_block(x_retain)

        x_pred = self.x_input_pred_block(x)
        stm_pred = self.stm_input_pred_block(stm)
        x_pred = x_pred + stm_pred
        pred = self.final_pred_block(x_pred)

        ltm_pred = retain_wei * pred

        return ltm_pred
    
    
class Output_Gate(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self.x_input_retain_block = nn.Linear(input_size, hidden_size)
        self.stm_input_retain_block = nn.Linear(hidden_size, hidden_size)
        self.final_retain_block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        self.stm_pred_block = nn.Tanh()

    def forward(self,x,stm,ltm):

        x_retain = self.x_input_retain_block(x)
        stm_retain = self.stm_input_retain_block(stm)
        x_retain = x_retain + stm_retain
        retain_wei = self.final_retain_block(x_retain)

        stm_pred = self.stm_pred_block(ltm)
        stm_pred = stm_pred * retain_wei

        return stm_pred
    


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, batch_size = 1, num_layers = 1, dropout = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.layers = nn.ModuleList()

        for layer in range(num_layers):

            layer_input_size = input_size if layer == 0 else hidden_size
            self.layers.append(
                nn.ModuleDict({
                'forget_gate': Forget_Gate(layer_input_size, hidden_size),
                'update_gate': Update_Gate(layer_input_size, hidden_size),
                'output_gate': Output_Gate(layer_input_size, hidden_size)
            }))

        self.prediction = nn.Linear(hidden_size, output_size)

    def forward(self, x_t, hidden = None):
        
        
        if hidden is None:
            stm = [torch.zeros(1, self.hidden_size, device = device) for _ in range(self.num_layers)]
            ltm = [torch.zeros(1, self.hidden_size, device = device) for _ in range(self.num_layers)]
        else:
            stm, ltm = hidden

        layer_input = x_t
        new_stm = []
        new_ltm = []

        for layer in range(self.num_layers):
            
            forget = self.layers[layer]['forget_gate']
            update = self.layers[layer]['update_gate']
            output = self.layers[layer]['output_gate']


            ltm_forget_wei = forget(layer_input, stm[layer])
            ltm[layer] = ltm[layer] * ltm_forget_wei

            ltm_update_bias = update(layer_input, stm[layer])
            ltm[layer] = ltm[layer] + ltm_update_bias

            stm[layer] = output(layer_input, stm[layer], ltm[layer])

            if layer < self.num_layers - 1 and self.training:
                layer_input = self.dropout(stm[layer])

            else:
                layer_input = stm[layer]

            new_stm.append(stm[layer])
            new_ltm.append(ltm[layer])

        output = self.prediction(new_stm[-1])

        return output, (new_stm, new_ltm)

    def evaluate(self, x):
        
        # x = (batch_size, seq_len, input_size)
               
        outputs = torch.zeros(batch_size, seq_len, output_size, device = device)

        for batch_no in range(batch_size):

            batch_seq = x[batch_no]  # Shape: (seq_len, input_size)
            hidden = None
            batch_outputs = []

            for t in range(seq_len):

                x_t = batch_seq[t].unsqueeze(0)  # Shape: (1, input_size)
                output, hidden = self.forward(x_t, hidden)
                batch_outputs.append(output)

            batch_outputs = torch.cat(batch_outputs, dim=0)  # Shape: (seq_len, output_size)

            outputs[batch_no] = batch_outputs
            
        return outputs



# setting up the model and optimizer
model = LSTM(input_size, hidden_size, output_size, batch_size, num_layers, dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# initializing the weights of the model as per pytorch documentation of the LSTM module
def weights_init(model):
    k = np.sqrt(1 / hidden_size)
    if isinstance(model, nn.Linear):
        nn.init.uniform_(model.weight, -k, k)
        nn.init.uniform_(model.bias, -k, k)

model.apply(weights_init)

test_losses = []
train_losses = []

epochs = []

# training the model
for epoch in range(training_epochs):

    model.train()

    if epoch % eval_interval == 0:

        model.eval()

        train_loss = calculate_loss()
        test_loss = calculate_loss("test")
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        epochs.append(epoch)

        model.train()

    # loss = calculate_loss()

      # Get a batch
    x, y = get_batch()
    
    # Forward pass using evaluate method which processes the whole sequence
    y_pred = model.evaluate(x)
    
    # Calculate loss
    lossfn = RMSEloss()
    loss = lossfn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# plotting the training results
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, test_losses, label="Test Loss")
plt.title("Training vs Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend()  
plt.show()