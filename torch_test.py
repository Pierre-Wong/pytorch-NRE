import torch
from torch.autograd import Variable

import numpy as np
import time
import datetime
import os
#from sklearn.metrics import average_precision_score
from relation import remodel
import copy

class Settings(object):

	def __init__(self):

		self.vocab_size = 114042
		self.num_steps = 70
		self.num_epochs = 50
		self.num_classes = 53
		self.gru_size = 230
		self.keep_prob = 0.5
		self.num_layers = 1
		self.pos_size = 5
		self.pos_num = 123
		# the number of entity pairs of each batch during training or testing
        self.big_num = 50



if __name__ == "__main__":
    # ATTENTION: change pathname before you load your model
    

    test_settings =Settings()
    test_settings.vocab_size = 114044
    test_settings.num_classes = 53
    test_settings.big_num =20

    big_num_test = test_settings.big_num



    def test_step(word_batch, pos1_batch, pos2_batch, y_batch,mtest):
        
        #print('total len=',num)
        #feed_dict = {}
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
        total_word=Variable(torch.from_numpy(total_word),volatile=True).cuda()
        total_pos1=Variable(torch.from_numpy(total_pos1),volatile=True).cuda()
        total_pos2=Variable(torch.from_numpy(total_pos2),volatile=True).cuda()
        
        

        _,acc,prob=mtest(total_word, total_pos1, total_pos2, total_shape, y_batch)
        
        accuracy=np.array(acc)
        prob=np.array(prob)
       
        return prob, accuracy

    # evaluate p@n
    def eval_pn(test_y, test_word, test_pos1, test_pos2, test_settings,mtest):
        allprob = []
        acc = []
        #print('len testword=',len(test_word))
        for i in range(int(len(test_word) / float(test_settings.big_num))):
            tempw=test_word[i * test_settings.big_num:(i + 1) *test_settings.big_num]
            temp1=test_pos1[i * test_settings.big_num:(i + 1) *test_settings.big_num]
            temp2=test_pos2[i * test_settings.big_num:(i + 1) *test_settings.big_num]
            tempy= test_y[i * test_settings.big_num:(i + 1) * test_settings.big_num]
            prob, accuracy = test_step(tempw,temp1,temp2,tempy,mtest)

            acc.append(np.mean(np.reshape(np.array(accuracy),(test_settings.big_num))))
            prob = np.reshape(np.array(prob), (test_settings.big_num, test_settings.num_classes))
            for single_prob in prob:
                allprob.append(single_prob[1:])

        print('allshape=',np.array(allprob).shape)
        allprob = np.reshape(np.array(allprob), (-1))
        #import pickle
        #f=open('allprob.pkl','wb')

        #pickle.dump(allprob,f,-1)
        #f.close()
        eval_y = []
        for i in test_y:
            eval_y.append(i[1:])
        allans = np.reshape(eval_y, (-1))
        order = np.argsort(-allprob)
        
        print('len test_y=',len(test_y))
        print('P@100:')
        top100 = order[:100]
        correct_num_100 = 0.0
        for i in top100:
            if allans[i] == 1:
                correct_num_100 += 1.0
        print(correct_num_100 / 100)

        print('P@200:')
        top200 = order[:200]
        correct_num_200 = 0.0
        for i in top200:
            if allans[i] == 1:
                correct_num_200 += 1.0
        print(correct_num_200 / 200)

        print('P@300:')
        top300 = order[:300]
        correct_num_300 = 0.0
        for i in top300:
            if allans[i] == 1:
                correct_num_300 += 1.0
        print(correct_num_300 / 300)



    # ATTENTION: change the list to the iters you want to test !!
    # testlist = range(9025,14000,25)
    model_iter=10300
    #mtest=remodel(len(wordembedding[0]),230,test_settings.vocab_size,wordembedding,big_num_test)
    #mtest.cuda()
    path='./torch_model/model'+str(model_iter)+'.pth'
    mtest=torch.load(path)
    mtest.cuda()
    mtest.big_num=big_num_test
    mtest.eval()
    print("Evaluating P@N for iter " + str(model_iter))


    print('Evaluating P@N for one')


    test_y = np.load('./data/pone_test_y.npy')
    test_word = np.load('./data/pone_test_word.npy')
    test_pos1 = np.load('./data/pone_test_pos1.npy')
    test_pos2 = np.load('./data/pone_test_pos2.npy')
    eval_pn(test_y, test_word, test_pos1, test_pos2, test_settings,mtest)
    
   
    print('Evaluating P@N for two')

    test_y = np.load('./data/ptwo_test_y.npy')
    test_word = np.load('./data/ptwo_test_word.npy')
    test_pos1 = np.load('./data/ptwo_test_pos1.npy')
    test_pos2 = np.load('./data/ptwo_test_pos2.npy')
    eval_pn(test_y, test_word, test_pos1, test_pos2, test_settings,mtest)

    print('Evaluating P@N for all')

    test_y = np.load('./data/pall_test_y.npy')
    test_word = np.load('./data/pall_test_word.npy')
    test_pos1 = np.load('./data/pall_test_pos1.npy')
    test_pos2 = np.load('./data/pall_test_pos2.npy')
    eval_pn(test_y, test_word, test_pos1, test_pos2, test_settings,mtest)

    time_str = datetime.datetime.now().isoformat()
    print(time_str)
    print('Evaluating all test data and save data for PR curve')


    test_y = np.load('./data/testall_y.npy')
    test_word = np.load('./data/testall_word.npy')
    test_pos1 = np.load('./data/testall_pos1.npy')
    test_pos2 = np.load('./data/testall_pos2.npy')
    allprob = []
    acc = []
    for i in range(int(len(test_word) / float(test_settings.big_num))):
        tempw=test_word[i * test_settings.big_num:(i + 1) *test_settings.big_num]
        temp1=test_pos1[i * test_settings.big_num:(i + 1) *test_settings.big_num]
        temp2=test_pos2[i * test_settings.big_num:(i + 1) *test_settings.big_num]
        tempy= test_y[i * test_settings.big_num:(i + 1) * test_settings.big_num]
        prob, accuracy = test_step(tempw,temp1,temp2,tempy,mtest)

        acc.append(np.mean(np.reshape(np.array(accuracy), (test_settings.big_num))))
        prob = np.reshape(np.array(prob), (test_settings.big_num, test_settings.num_classes))
        for single_prob in prob:
            allprob.append(single_prob[1:])
    allprob = np.reshape(np.array(allprob), (-1))
    order = np.argsort(-allprob)

    print('saving all test result...')
    current_step = model_iter

    # ATTENTION: change the save path before you save your result !!
    #np.save('./out/allprob_iter_' + str(current_step) + '.npy', allprob)
    allans = np.load('./data/allans.npy')

    # caculate the pr curve area
    #average_precision = average_precision_score(allans, allprob)
    #print('PR curve area:' + str(average_precision))



    time_str = datetime.datetime.now().isoformat()
    print(time_str)
    print('P@N for all test data:')
    print('P@100:')
    top100 = order[:100]
    correct_num_100 = 0.0
    for i in top100:
        if allans[i] == 1:
            correct_num_100 += 1.0
    print(correct_num_100 / 100)

    print('P@200:')
    top200 = order[:200]
    correct_num_200 = 0.0
    for i in top200:
        if allans[i] == 1:
            correct_num_200 += 1.0
    print(correct_num_200 / 200)

    print('P@300:')
    top300 = order[:300]
    correct_num_300 = 0.0
    for i in top300:
        if allans[i] == 1:
            correct_num_300 += 1.0
    print(correct_num_300 / 300)





