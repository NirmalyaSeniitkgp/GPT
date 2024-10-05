import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import Positional_Encoding_Generation
from Positional_Encoding_Generation import PositionalEncoding
import GPT_Model_3
from GPT_Model_3 import GPT

device = 'cpu'

torch.autograd.set_detect_anomaly(True)

# Prepare English Vocabulary
vocabulary = ['\n', ' ', '!',  '$', '&', "'", ',', '-', '.',  '3', ':', ';',  '?',  'A', 
              'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
              'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e',
              'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
               'u', 'v', 'w', 'x', 'y', 'z']


# Create a Mapping from string to integers
string_to_integer = {v:k for k,v in enumerate(vocabulary)}
# Create a Mapping from integers to string
integer_to_string = {k:v for k,v in enumerate(vocabulary)}

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
validation_data_tokenize = data_tokenize[n::1]
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

# GPT Model Generation
language_to_index = string_to_integer
num_heads = 8
hidden = 2048
num_layers = 5
vocabulary_size = len(string_to_integer)

model = GPT(d_model,language_to_index,
            num_heads,hidden,num_layers,vocabulary_size).to(device)


#This is for Model Loading
model.load_state_dict(torch.load('/home/idrbt-06/Desktop/PY_TORCH/GPT/GPT_5_layers_300_epochs_batch_size_64_lr_0.0003.pth', map_location=torch.device(device)))
model.eval()
print(model)

# This is for Testing

def text_generation(tokens, max_new_tokens):
    for i in range(max_new_tokens):
        input_tokens = tokens[:,-block_size::1]
        output = model(input_tokens, position, mask)
        # Output is a 3D Tensor (batch_size, max_sequence_length, vocabulary_size)
        # We are considering only the last row
        # Because last row is the output which considers full block_size amount of input tokens
        output = output[:,-1,:]
        next_token = torch.argmax(output)
        next_token = torch.tensor([[next_token.item()]])
        #print(next_token)
        tokens = torch.cat((tokens,next_token),dim=1)
    return tokens


starting_tokens = validation_data_tokenize[0:block_size:1]
input_text = decode(starting_tokens.tolist())
print(input_text)
print('*'*100)

initial_tokens = starting_tokens.reshape(1,len(starting_tokens))
max_new_tokens = 1000

generated_tokens = text_generation(initial_tokens,max_new_tokens)
generated_tokens = generated_tokens.tolist()[0]

generated_text = decode(generated_tokens)
print(generated_text)
print('*'*100)



