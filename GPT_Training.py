import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import Positional_Encoding_Generation
from Positional_Encoding_Generation import PositionalEncoding
import GPT_Model_3
from GPT_Model_3 import GPT


batch_size = 64
learning_rate = 0.0003
num_epochs = 100

# Selecting the Device
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device = get_device()

#device = 'cuda'

torch.autograd.set_detect_anomaly(True)

# Prepare English Vocabulary
vocabulary = ['\n', ' ', '!',  '$', '&', "'", ',', '-', '.',  '3', ':', ';',  '?',  'A', 
              'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
              'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e',
              'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
               'u', 'v', 'w', 'x', 'y', 'z']


# Reading a File (This is tinyshakespears)
with open('input.txt','r',encoding='utf-8') as file:
    data = file.read()
#print(data)

# Create a Mapping from string to integers
string_to_integer = {v:k for k,v in enumerate(vocabulary)}
#print(string_to_integer)
# Create a Mapping from integers to string
integer_to_string = {k:v for k,v in enumerate(vocabulary)}
# Encoder Function
encode = lambda s:[string_to_integer[j] for j in s]
# Decoder Function
decode = lambda l:''.join([integer_to_string[p] for p in l])

# Tokenize the complete File of tinyshakespears and converts it to a tensor
data_tokenize = torch.tensor(encode(data),device='cpu')
print(data_tokenize)
print(len(data_tokenize))

# Split the data into train data and validation data
# First 90% will be train data and remaining 10% will be validation data
n = int(0.9*len(data))
train_data_tokenize = data_tokenize[0:n:1]
validation_data_tokenize = data_tokenize[n::1]
print('\n')
print('This is Length of Train Data Tokenize :', len(train_data_tokenize))
print('\n')
print('This is Length of Validation Data Tokenize :',len(validation_data_tokenize))

# Assignment of max_sequence_length and d_model
max_sequence_length = 256
d_model = 512
block_size = max_sequence_length

# Generation of Positional Information using sin and cos function
position_encoder = PositionalEncoding(max_sequence_length, d_model)
print('I am Here    1')
position = ((position_encoder.forward()).requires_grad_(False)).to(device)

# Generating the Causal Mask
mask = torch.tril(torch.ones((max_sequence_length, max_sequence_length),device = device))
print('I am Here    2')
mask[mask==0] = -torch.inf
mask[mask==1] = 0

# Generating the Inputs and Labels
inputs = []
labels = []

for i in range(0,(len(train_data_tokenize)-block_size),block_size):
    x = train_data_tokenize[i:i + block_size]
    y = train_data_tokenize[i+1:i+1+block_size]
    inputs.append(x)
    labels.append(y)

inputs_stacks = torch.stack(inputs)
labels_stacks = torch.stack(labels)
print('\n')
print(inputs_stacks.shape)
print('\n')
print(labels_stacks.shape)


# Preparing the Dataset
class Our_Dataset(Dataset):
    def __init__(self, inputs_stacks: Tensor, labels_stacks: Tensor)-> tuple[Tensor]:
        self.inputs_stacks = inputs_stacks
        self.labels_stacks = labels_stacks
    
    def __len__(self):
        total_number_of_input_sequences = len(self.inputs_stacks)
        return total_number_of_input_sequences
    
    def __getitem__(self, index):
        data_point = (self.inputs_stacks[index], self.labels_stacks[index])
        return data_point

text_dataset = Our_Dataset(inputs_stacks, labels_stacks)

# Preparing the Data Loader
train_loader = DataLoader(dataset = text_dataset, batch_size = batch_size, drop_last = True)

# GPT Model Generation
language_to_index = string_to_integer
num_heads = 8
hidden = 2048
num_layers = 6
vocabulary_size = len(string_to_integer)

model = GPT(d_model,language_to_index,
            num_heads,hidden,num_layers,vocabulary_size).to(device)

# Initialization of GPT model
for params in model.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)

# Printing the Detail Architecture of Model
print(model)

# # Using more than one GPU
# if torch.cuda.device_count()>1:
#     model = nn.DataParallel(model)

# Defining the Loss Function
criterion = nn.CrossEntropyLoss()
# Defining the Optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, betas=(0.9,0.98))

for epoch in range(num_epochs):
    for batch_number, (samples, labels) in enumerate(train_loader):
        print('This is batch_number = ', batch_number)
        print('This is Sample Size = ', samples.shape)
        print('This is Label Size =', labels.shape)

        # samples are the Inputs. It is a 2D Tensor.
        samples = samples.to(device)
        # labels are the Targets. It is a 2D Tensor.
        labels = labels.to(device)

        print(samples)
        print(labels)

        # Erase the Previously Calculated Gradients
        optimizer.zero_grad()

        # Calculation of Model Output
        # Model Output is a 3D Tensor
        output = model(samples, position, mask)
        print(output.shape)

        # Convert the Model Output to a 2D Tensor
        output = output.reshape(batch_size*max_sequence_length, vocabulary_size)
        print(output.shape)

        # Convert the Label to a 1D Tensor
        labels = labels.reshape(batch_size*max_sequence_length)

        # Calculate Loss
        loss = criterion(output, labels)

        # Backward Pass and Optimization
        # Calculate the Gradients
        loss.backward()

        # Clip the Gradients values if they are less than -0.9 or they are more than +0.9
        # nn.utils.clip_grad.clip_grad_value_(model.parameters(), 0.9)

        # Making the Global Gradients to Unit Norm
        nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.0)

        # Update the Model Parameters
        # Complete Single Optimization Step
        optimizer.step()


# This is for Model Saving
torch.save(model.state_dict(), 'GPT_6_layers_100_epochs_batch_size_64_lr_0.0003.pth')



