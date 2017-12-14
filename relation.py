import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
from torch import autograd
import pickle
from torch.autograd import Variable
import numpy as np
import string
import random as rd  # 导入random模块，使用里面的sample函数
import copy
import torch.utils.data as Data


class remodel(nn.Module):
        def __init__(self, embedding_dim, hidden_dim, vocab_size, pre_emb,big_num):
            super(remodel, self).__init__()
            self.big_num=big_num
            self.hidden_dim = hidden_dim

            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.word_embeddings.weight = nn.Parameter(torch.from_numpy(pre_emb))
            
            self.pos1_embeddings=nn.Embedding(123,5)
            #nn.init.xavier_uniform(self.pos1_embeddings.weight)
            nn.init.normal(self.pos1_embeddings.weight)
            self.pos2_embeddings=nn.Embedding(123,5)
            nn.init.normal(self.pos2_embeddings.weight)
            #nn.init.xavier_uniform(self.pos2_embeddings.weight)


            self.lstm = nn.GRU(embedding_dim+10, hidden_dim,bidirectional=True,dropout=0.2,batch_first=True)
            #self.bilinear=nn.Linear(hidden_dim*2,hidden_dim)

            self.attention_w=Variable(torch.FloatTensor(hidden_dim,1),requires_grad=True).cuda()
            nn.init.xavier_uniform(self.attention_w)

            self.sen_a=Variable(torch.FloatTensor(hidden_dim),requires_grad=True).cuda()
            nn.init.normal(self.sen_a)
            #nn.init.xavier_uniform(self.sen_a)

               
            self.sen_r=Variable(torch.FloatTensor(hidden_dim,1),requires_grad=True).cuda()
            nn.init.normal(self.sen_r)
            #nn.init.xavier_uniform(self.sen_r)


            self.relation_embedding=Variable(torch.FloatTensor(53,hidden_dim),requires_grad=True).cuda()
            nn.init.normal(self.relation_embedding)
            #nn.init.xavier_uniform(self.relation_embedding)

            self.sen_d = Variable(torch.FloatTensor( 53), requires_grad=True).cuda()
            nn.init.normal(self.sen_d)
            #nn.init.xavier_uniform(self.sen_d)

            self.ls=nn.BCEWithLogitsLoss()
        
        def forward(self, sentence,pos1,pos2,total_shape,y_batch):
            total_num = total_shape[-1]
            #print('totalnum=',total_num)
            #self.hidden = self.init_hidden(total_num)
            embeds1 = self.word_embeddings(sentence)
            pos1_emb=self.pos1_embeddings(pos1)
            pos2_emb=self.pos2_embeddings(pos2)
            inputs=torch.cat([embeds1,pos1_emb,pos2_emb],2)

            tup,_ = self.lstm(
                inputs)
            
            tupf=tup[:,:,range(self.hidden_dim)]
            tupb=tup[:,:,range(self.hidden_dim,self.hidden_dim*2)]
            tup=torch.add(tupf,tupb)
            tup=tup.contiguous()
            
            tup1=F.tanh(tup).view(-1,self.hidden_dim)
            
           
            tup1=torch.matmul(tup1,self.attention_w).view(-1,70)
            
            tup1=F.softmax(tup1).view(-1,1,70)
            attention_r=torch.matmul(tup1,tup).view(-1,self.hidden_dim)
            sen_repre = []
            sen_alpha = []
            sen_s = []
            sen_out = []
            self.loss=[]
            self.prob=[]
            self.prob2=[]
            self.predictions=[]
            self.acc=[]
           
            for i in range(self.big_num):
                sen_repre.append(F.tanh(attention_r[total_shape[i]:total_shape[i + 1]]))
                
                batch_size = total_shape[i + 1] - total_shape[i]

                
             
                sen_alpha.append(F.softmax(torch.matmul(torch.mul(sen_repre[i],self.sen_a),self.sen_r).view(batch_size)).view(1,batch_size))
        
                sen_s.append(torch.matmul(sen_alpha[i],sen_repre[i]).view(self.hidden_dim,1))


                sen_out.append(torch.add(torch.matmul(self.relation_embedding,sen_s[i]).view(53),self.sen_d))
                
                
                self.prob.append(F.softmax(sen_out[i]))
                self.prob2.append(F.softmax(sen_out[i]).cpu().data.numpy())
                
                _,pre=torch.max(self.prob[i],0)
                self.predictions.append(pre)
                
 
                self.loss.append(torch.mean(self.ls(sen_out[i],Variable(torch.from_numpy(y_batch[i].astype(np.float32))).cuda())))
                 
                
                if i == 0:
                    self.total_loss = self.loss[i]
                else:
                    self.total_loss += self.loss[i]
          
                s=np.mean(np.equal(pre.cpu().data.numpy(),np.argmax(y_batch[i])).astype(float))
                
                self.acc.append(s)
                
            return self.total_loss,self.acc,self.prob2

if __name__ == '__main__':
    torch.manual_seed(13)  # 设定随机数种子
    BATCH_SIZE = 64


         
          
               
                
                

    print('reading wordembedding')
    wordembedding = np.load('./data/vec.npy')

    print('reading training data')
    train_y = np.load('./data/small_y.npy')
    train_word = np.load('./data/small_word.npy')
    train_pos1 = np.load('./data/small_pos1.npy')
    train_pos2 = np.load('./data/small_pos2.npy')

    vocab_size = len(wordembedding)
    num_classes = len(train_y[0])

    big_num = 50

    model=remodel(len(wordembedding[0]),230,vocab_size,wordembedding,big_num)
    model.cuda()
    model.train() 
    optimizer=optim.Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
    def train_step(model,word_batch, pos1_batch, pos2_batch, y_batch, big_num,steps):
        optimizer.zero_grad()
        total_shape = []
        total_num = 0
        total_word = []
        total_pos1 = []
        total_pos2 = []
        for i in range(len(word_batch)):
            total_shape.append(total_num)
            total_num += len(word_batch[i])
            for word in word_batch[i]:
                total_word.append(word)
            for pos1 in pos1_batch[i]:
                total_pos1.append(pos1)
            for pos2 in pos2_batch[i]:
                total_pos2.append(pos2)
        total_shape.append(total_num)
        #total_shape = np.array(total_shape)
        total_word = np.array(total_word)
        total_pos1 = np.array(total_pos1)
        total_pos2 = np.array(total_pos2)
        total_word=Variable(torch.from_numpy(total_word)).cuda()
        total_pos1=Variable(torch.from_numpy(total_pos1)).cuda()
        total_pos2=Variable(torch.from_numpy(total_pos2)).cuda()
        loss,acc,_=model(total_word,total_pos1,total_pos2,total_shape,y_batch)
        loss.backward()
        acc=np.reshape(np.array(acc),(big_num))
        acc=np.mean(acc)
        steps+=1
        if  steps>=8000 and steps%50==0:
           #print('model.acc=',acc)
           print('steps=',steps)
           print('acc=',acc)
           #print('y=',y_batch)
           print('loss=',loss)
           path='./torch_model/model'+str(steps)+'.pth'
           torch.save(model, path)    
           #a=input()
        optimizer.step()
        return acc,loss,steps





    steps=0
    for epoch in range(3):
        temp_order = list(range(len(train_word)))
       
        np.random.shuffle(temp_order)
        #print('temp_order=',temp_order)
        #a=input()
        for i in range(int(len(temp_order) / float(big_num))):

            temp_word = []
            temp_pos1 = []
            temp_pos2 = []
            temp_y = []

            temp_input = temp_order[i * big_num:(i + 1) * big_num]
            for k in temp_input:
                temp_word.append(train_word[k])
                temp_pos1.append(train_pos1[k])
                temp_pos2.append(train_pos2[k])
                temp_y.append(train_y[k])
            num = 0
            for single_word in temp_word:
                num += len(single_word)

            if num > 1500:
                print('out of range')
                continue

            temp_word = np.array(temp_word)
            temp_pos1 = np.array(temp_pos1)
            temp_pos2 = np.array(temp_pos2)
            temp_y = np.array(temp_y)
            
            
            #print('start')
            acc,loss,steps=train_step(model,temp_word, temp_pos1, temp_pos2, temp_y, big_num,steps)
        print('acc=',acc)







