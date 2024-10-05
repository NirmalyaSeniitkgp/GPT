import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math


# Selecting the Device
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
device = get_device()

#device = 'cuda'

# Calculation of Soft Max Probability
# This function is neumerically more stable than PyTorch softmax function
def nirmalya_softmax(input_tensor):
    eps = 1e-10
    p1 = torch.exp(input_tensor)
    p2 = p1.sum(dim=-1, keepdim=True) + eps
    output_tensor = p1/p2
    return output_tensor

# Calculation of Embedding and adding with Positional Encoding
class SentenceEmbedding(nn.Module):
    def __init__(self,language_to_index,d_model):
        super().__init__()
        self.language_to_index = language_to_index
        self.d_model = d_model
        self.vocabulary_size = len(self.language_to_index)
        self.embedding = nn.Embedding(self.vocabulary_size, d_model)
        self.dropout = nn.Dropout(p = 0.1)
    
    def forward(self, x, position):
        x = ((self.embedding(x))*(math.sqrt(self.d_model))).to(device)
        x = self.dropout(x + position)
        return x

# Calculation of Scaled Dot Product Attention
def Scaled_Dot_Product_Attention(q, k, v, mask):
    d_k = q.shape[-1]
    scaled_dot_product = (torch.matmul(q, k.transpose(-2,-1))/math.sqrt(d_k))
    #print('I am Here')
    scaled_dot_product = scaled_dot_product + mask
    attention = nirmalya_softmax(scaled_dot_product)
    out = torch.matmul(attention, v)
    return attention, out

# Calculation of Multi Head Attention
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model//num_heads
        self.q_layer = nn.Linear(d_model, d_model, bias = False)
        self.k_layer = nn.Linear(d_model, d_model, bias = False)
        self.v_layer = nn.Linear(d_model, d_model, bias = False)
        self.linear_layer = nn.Linear(d_model, d_model, bias = False)
    
    def forward(self, x, mask):
        batch_size, max_sequence_length, d_model = x.shape
        #print(f'Size of input tensor: {batch_size, max_sequence_length, d_model}')

        # Generation of q Tensor
        q = self.q_layer(x)
        #print(q.shape)
        q = q.reshape(batch_size, max_sequence_length, self.num_heads, self.head_dim)
        #print(q.shape)
        q = q.permute(0, 2, 1, 3)
        # Now the Shape will be (batch_size, num_heads, max_sequence_length, head_dim)
        #print(q.shape)
        #print('**')

        # Generation of k Tensor
        k = self.k_layer(x)
        #print(k.shape)
        k = k.reshape(batch_size, max_sequence_length, self.num_heads, self.head_dim)
        #print(k.shape)
        k = k.permute(0, 2, 1, 3)
        # Now the Shape will be (batch_size, num_heads, max_sequence_length, head_dim)
        #print(k.shape)
        #print('***')

        # Generation of v Tensor
        v = self.v_layer(x)
        #print(v.shape)
        v = v.reshape(batch_size, max_sequence_length, self.num_heads, self.head_dim)
        #print(v.shape)
        v = v.permute(0, 2, 1, 3)
        # Now the Shape will be (batch_size, num_heads, max_sequence_length, head_dim)
        #print(v.shape)
        #print('V\n',v)
        #print('****')

        # Calculation of Multi Head Attention
        (attention, attention_head)= Scaled_Dot_Product_Attention(q, k, v, mask)
        # Shape of Attention Probability will be (batch_size, num_heads, max_sequence_length, max_sequence_length)
        #print(attention.shape)
        # Shape of Attention Head will be (batch_size, num_heads, max_sequence_length, head_dim)
        #print(attention_head.shape)
        #print('Attention\n',attention)
        #print('Attention Head\n',attention_head)

        # Concatination of Multiple Heads
        attention_head = attention_head.permute(0, 2, 1, 3)
        # Now the Shape of Attention Head will be (batch_size, max_sequence_length, num_heads, head_dim)
        #print(attention_head.shape)
        #print(attention_head)
        attention_head = attention_head.reshape(batch_size, max_sequence_length, self.num_heads*self.head_dim)
        # Now the Shape of Attention Head will be (batch_size, max_sequence_length, d_model)
        #print(attention_head.shape)
        #print(attention_head)

        # Inter Communication between Multiple Heads
        z = self.linear_layer(attention_head)
        # Shape of z tensor will be (batch_size, max_sequence_length, d_model)
        #print(z.shape)
        return z

# Calculation of Layer Normalization
class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape):
        super(LayerNormalization, self).__init__()
        self.parameters_shape = parameters_shape
        self.eps = 0.00001
        self.gamma = nn.Parameter(torch.ones(parameters_shape, device = device))
        self.beta = nn.Parameter(torch.zeros(parameters_shape, device = device))
    
    def forward(self, x):
        if len(self.parameters_shape)==1:
            dims = [-1]
        else:
            dims = [-2, -1]
        mean_values = (x.mean(dim=dims, keepdim=True))
        #print(mean_values)
        #print(mean_values.shape)
        #print('I am Here    4')
        variance_values = (x.var(dim=dims, unbiased=False, keepdim=True))
        #print(variance_values)
        #print(variance_values.shape)
        out = (((x-mean_values)/torch.sqrt(variance_values + self.eps))*self.gamma + self.beta)
        return out

# Application of Position Wise Feed Forward Network
class PositionwiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, hidden, drop_prob = 0.1) -> Tensor:
        super(PositionwiseFeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(in_features = d_model, out_features = hidden)
        self.linear2 = nn.Linear(in_features = hidden, out_features = d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = drop_prob)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# Implementation of Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden, drop_prob = 0.1):
        super(DecoderLayer, self).__init__()
        self.attention = MultiheadAttention(d_model=d_model,num_heads=num_heads)
        self.dropout1 = nn.Dropout(p = drop_prob)
        self.norm1 = LayerNormalization(parameters_shape = [d_model])
        self.pffn = PositionwiseFeedForwardNetwork(d_model=d_model,hidden=hidden,drop_prob=drop_prob)
        self.dropout2 = nn.Dropout(p = drop_prob)
        self.norm2 = LayerNormalization(parameters_shape = [d_model])

    def forward(self, x, mask):
        residual_x = x.clone()
        x = self.attention(x, mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x.clone()
        x = self.pffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x

# Implementation of Sequential Decoder Layers
class SequentialDecoderLayersGeneration(nn.Sequential):
    def forward(self, x, mask):
        for i in self._modules.values():
            x = i(x, mask)
        return x

# Implementation of GPT
class GPT(nn.Module):
    def __init__(self,
                d_model,
                language_to_index,
                num_heads,
                hidden,
                num_layers,
                vocabulary_size) -> Tensor:
        super(GPT,self).__init__()
        self.sentance_embedding = SentenceEmbedding(language_to_index,d_model)
        self.layers = SequentialDecoderLayersGeneration(*[DecoderLayer(d_model, num_heads, hidden, drop_prob = 0.1)
                                                        for j in range(num_layers)])
        self.linear = nn.Linear(in_features = d_model, out_features = vocabulary_size)

    def forward(self, x, position, mask):
        x = self.sentance_embedding(x, position)
        x = self.layers(x, mask)
        x = self.linear(x)
        return x


