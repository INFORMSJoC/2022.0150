import pickle
import matplotlib.pyplot as plt
import os
'''
Experimental details:
    Model     : WRN
    Optimizer : sgd
    Momentum : 0.5
    Learning  : 0.01
    Weight_decay  : 0.0005
    Global Rounds   : 10000
    Total clients   : 250

    Federated parameters:
    IID
    Fraction of users  : 0.04
    Local Epochs       : 8
    Local Batch size   : 50
    
    factor 0.2  +30

'''
print(os.getcwd())

base9 = pickle.load(open('./save/' + 'cifar100_WRN_method[fedavg]_linear1.pkl', 'rb'))
train_loss_b9, train_accuracy_b9, test_accuracy_b9, interacts_list_b9 = base9[0], base9[1], base9[2], base9[3]

ascend = pickle.load(open('./save/' + 'cifar100_WRN_method[ascend]_linear1.pkl', 'rb'))
train_loss_ascend, train_accuracy_ascend, test_accuracy_ascend, interacts_list_ascend = ascend[0], ascend[1], ascend[2], \
                                                                                        ascend[3]

cl = pickle.load(open('./save/' + 'cifar100_WRN_method[fedecs]_linear1.pkl', 'rb'))
train_loss_cl, train_accuracy_cl, test_accuracy_cl, interacts_list_cl = cl[0], cl[1], cl[2], cl[3] 

anticl = pickle.load(open('./save/' + 'cifar100_WRN_method[afl]_linear1.pkl', 'rb'))
train_loss_anticl, train_accuracy_anticl, test_accuracy_anticl, interacts_list_anticl = anticl[0], anticl[1], anticl[2], \
                                                                                        anticl[3]

other =  pickle.load(open('./save/' + 'cifar100_WRN_method[favor]_linear1.pkl', 'rb'))      
train_loss_other, train_accuracy_other, test_accuracy_other, interacts_list_other = other[0], other[1], other[2], \
                                                                                        other[3]                                                                         

plt.figure()
plt.clf() 
plt.xlabel('interaction times')
plt.ylabel('test accuracy')
# plt.subplots_adjust(bottom=0.16,left=0.18,right=0.98, top=0.98)

plt.plot([sum(interacts_list_b9[0:i + 1]) for i in range(len(test_accuracy_b9))][::15], test_accuracy_b9[::15],
        ls='--',marker='<',markevery=2,linewidth = 2, 
        label='FedAvg')
plt.plot([sum(interacts_list_ascend[0:i + 1]) for i in range(len(test_accuracy_ascend))][::15], test_accuracy_ascend[::15],
        ls='--', marker='*',markevery=2,linewidth = 2, 
        label='ASCEND')

plt.plot([sum(interacts_list_anticl[0:i + 1]) for i in range(len(test_accuracy_anticl))][::15], test_accuracy_anticl[::15], 
         ls='--', marker='.',markevery=2,linewidth = 2, 
         label='AFL')

plt.plot([sum(interacts_list_other[0:i + 1]) for i in range(len(test_accuracy_other))][::15], test_accuracy_other[::15], 
         ls='--', marker='+',markevery=2,linewidth = 2, 
         label='FAVOR')
plt.plot([sum(interacts_list_cl[0:i + 1]) for i in range(len(test_accuracy_cl))][::15], test_accuracy_cl[::15],
        ls='--',marker='s',markevery=2,linewidth = 2, 
        label='FedECS')
plt.grid(ls='-.')
plt.legend(loc=4)

plt.show()
