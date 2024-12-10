import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec, KeyedVectors
import string
import matplotlib.pyplot as plt
import math
import re
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

sentence_size = 128
word_vector_size = 128
word_vector_window = 5
epochs = 16
learning_rate = 0.000035 
batch_size = 64
hidden_size = 1024
n_layers=5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_df = pd.read_csv('train.csv')
train_df.head()
test_df = pd.read_csv('test.csv')
test_df.head()
train_df.drop(['id','keyword','location'],axis=1, inplace = True)
train_df.head()
test_df.drop(['keyword','location'],axis=1, inplace = True)
test_df.head()
train_raw_x = train_df['text']
test_raw_x = test_df['text']
def processText(text, target_length=None):
    
    start_token='<start>'
    end_token = '<end>'
    
    #text = ''.join(char for char in text.lower() if char not in string.punctuation)   #Make all lower case and remove punctuations
    text = text.lower()
    
    text = re.sub(r'[^\w\s]','',text) # remove punctuation
    text = re.sub(" \d+", " ", text) # remove pure number strings
    text = re.sub(r'http\S+','', text)
    
    
    tokens = word_tokenize(text)
    
    stopword =  stopwords.words('english')
    
    tokens =  [token for token in tokens if token not in stopword]
    
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens =  [lemmatizer.lemmatize(token) for token in tokens]
    
    lemmatized_tokens.insert(0,start_token)
    lemmatized_tokens.append(end_token)
    
    if target_length == None:
        return lemmatized_tokens
    
    while len(lemmatized_tokens) < target_length:
        lemmatized_tokens.extend(lemmatized_tokens)
    
    return lemmatized_tokens[0:target_length]
processed_trained_x = [processText(sentence,sentence_size) for sentence in train_raw_x]
processed_test_x = [processText(sentence,sentence_size) for sentence in test_raw_x]
all_x =processed_trained_x + processed_test_x
vector_model = Word2Vec(all_x,min_count=1,vector_size=word_vector_size,window=word_vector_window)
vector =  vector_model.wv
vector_train_x = []
for sentence in processed_trained_x:
    vector_train_x.append([vector[token].tolist() for token in sentence])
vector_test_x = []
for sentence in processed_test_x:
    vector_test_x.append([vector[token].tolist() for token in sentence])
x_train , y_train = vector_train_x , train_df['target']
class NLPData(Dataset):
    def __init__(self):
        self.x_data = torch.tensor(x_train ) #vector_train_x)
        self.y_data = torch.tensor(list(y_train),dtype=torch.float32)
        self.n_samples =  len(self.y_data)
    
    def __getitem__(self,idx):
        return self.x_data[idx] , self.y_data[idx]
    
    def __len__(self):
        return self.n_samples
dataset = NLPData()
train_loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True )
class LSTMNN(nn.Module):
    def __init__(self):
        super(LSTMNN,self).__init__()
        
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(word_vector_size,hidden_size,n_layers,batch_first=True)
        self.fc1 = nn.Linear(hidden_size,128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(128,32)
        self.fc3 = nn.Linear(32,1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0,c0))
        out = out[:,-1,:]
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        
        return out

model = LSTMNN().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.RMSprop(model.parameters(),lr=learning_rate)
all_loss =[]
for epoch in range(epochs):
    for x,y in train_loader:
        x,y = x.to(device), y.to(device).view(-1,1) 
        
        y_hat =  model(x)
        
        loss = criterion(y_hat,y)
        
        if(loss.item()<0.2): 
            break
        all_loss.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch: {epoch+1} Loss: {loss.item()}')
    if(loss.item()<0.2): 
            break
model.eval()
x_test = torch.tensor(vector_test_x).to(device)
#x_validation = torch.tensor(x_validation).to(device)
#y_pred = model(x_validation)
y_pred = model(x_test)
with torch.no_grad():
    y_pred =  np.round(y_pred.to('cpu').numpy()).astype(np.int32)
test_df.drop(['text'],axis=1,inplace=True)
test_df['target']=y_pred
test_df.to_csv('output.csv',index=False)