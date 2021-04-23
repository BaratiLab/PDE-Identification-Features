# importing the libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt
# for creating validation set
from sklearn.model_selection import train_test_split
# for evaluating the model
from sklearn.metrics import accuracy_score
# PyTorch libraries and modules
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
import ipdb
from sklearn.utils import shuffle
import pickle
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from itertools import groupby
import matplotlib.pyplot as plt
from pylab import * # For adjusting frame width only

from network import CNNModel

############### Global Parameters ###############
#video_path = './data/videos'
video_data_path='../labels.csv'
video_sub_path = './3DCNN_DATA/'
family = 't_b'
test_term = 'tt'
save_path = "./RESULTS/"+test_term+"_"+family+".csv"
#vgg16_path = '/home/cmu/Documents/tensorflow_vgg16/vgg16.tfmodel'
model_path = './models/'
############## Train Parameters #################
dim_image = 21
num_frames = 21
n_epochs = 10
batch_size = 12
learning_rate = 0.0017
num_classes = 2
##################################################

def split_text(s):
    for k, g in groupby(s, str.isalpha):
        yield ''.join(g)

def get_video_data(video_data_path, train_ratio=0.8):
    # Data import
    df = pd.read_csv (video_data_path)
    video_id = df['Video_ID']
    y_tt = df['label1'].to_numpy().reshape(-1,1)
    y_t = df['label2'].to_numpy().reshape(-1,1)
    y_x = df['label3'].to_numpy().reshape(-1,1)
    y_class = df['label_t'].to_numpy().reshape(-1,1)
    randomize = np.arange(len(video_id))
    np.random.shuffle(randomize)
    video_id = video_id[randomize]
    y_t = y_t[randomize]
    y_tt = y_tt[randomize]
    y_x = y_x[randomize]
    y_class = y_class[randomize]

    #### EXPERIMENT 1: All classes
    # dataset = y_class
    # kf = KFold(n_splits= 5 , shuffle=True)
    # for train_index, test_index in kf.split(dataset):  
    #     y_train = dataset[train_index]
    #     y_test = dataset[test_index]
    #     video_id_train = video_id[train_index]
    #     video_id_test = video_id[test_index]
    # train_size = int(0.8*8*384)
    # if test_term == "tt":
    #     y_train = y_tt[:train_size].reshape(-1)
    #     y_test = y_tt[train_size:].reshape(-1)
    # elif test_term == "t":
    #     y_train = y_t[:train_size].reshape(-1)
    #     y_test = y_t[train_size:].reshape(-1)
    # elif test_term == "x":
    #     y_train = y_x[:train_size].reshape(-1)
    #     y_test = y_x[train_size:].reshape(-1)

    # video_id_train = video_id[:train_size]
    # video_id_test = video_id[train_size:]

    # EXPERIMENT2: Selecting a family as test set
    ### FAMILY 
    if test_term == "tt":
        y_train = y_tt[video_id.str.startswith(family) == False].reshape(-1)
        y_test = y_tt[video_id.str.startswith(family)].reshape(-1)
    elif test_term == "t":
        y_train = y_t[video_id.str.startswith(family) == False].reshape(-1)
        y_test = y_t[video_id.str.startswith(family)].reshape(-1)
    elif test_term == "x":
        y_train = y_x[video_id.str.startswith(family) == False].reshape(-1)
        y_test = y_x[video_id.str.startswith(family)].reshape(-1)

    video_id_train = video_id[video_id.str.startswith(family) == False]
    video_id_test = video_id[video_id.str.startswith(family)]

    train_data = {'Video_ID': video_id_train , 'label': y_train}
    train_data = pd.DataFrame(data=train_data)
    test_data = {'Video_ID': video_id_test , 'label': y_test}
    test_data = pd.DataFrame(data=test_data)
    return train_data , test_data


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = CNNModel()
    net.to(device)
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(net.parameters(), lr= learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=[100,200,300,600,900,1200], gamma=0.8)
    train_data , test_data = get_video_data(video_data_path)
    train_loss = []
    test_loss = []
    for epoch in tqdm(range(n_epochs)):
        net.train()
        current_train_data =  train_data.sample(frac=1).reset_index(drop=True)
        running_loss = 0.0
        loss_epoch = 0
        loss_epoch_count = 0
        iteration = 0
        for start,end in zip(
                range(0, len(current_train_data), batch_size),
                range(batch_size, len(current_train_data)+1, batch_size)):
            iteration +=1
            current_batch = current_train_data[start:end]
            ### Based on the current batch videos, first get the type of equation (e.g. heat) then loads heat, and then gets the row (video number) from the file.
            current_names = current_batch['Video_ID'].values
            current_videos=[]
            for i in range(batch_size):
                data = np.load(video_sub_path + current_names[i] + '.npy' , allow_pickle = True)
                current_video = data.reshape(1,num_frames,dim_image,dim_image)
                current_videos.append(current_video)
            current_videos = np.array(current_videos)
            current_labels = current_batch['label'].values
            current_videos = torch.from_numpy(current_videos).float()
            current_labels = torch.from_numpy(current_labels).long()
            inputs, labels = current_videos.to(device) , current_labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if iteration % 10 == 0:   
                print('[%d, %5d] Train loss: %.3f' %
                    (epoch + 1, iteration , running_loss / 10))
                loss_epoch += running_loss / 10
                loss_epoch_count += 1 
                running_loss = 0.0
        scheduler.step()

        if np.mod(epoch + 1, 1) == 0:
            current_epoch_loss = loss_epoch / loss_epoch_count
            train_loss.append(current_epoch_loss)
            print ("Epoch ", epoch+1, " is done. Saving the model ...")
            torch.save(net.state_dict(), os.path.join(model_path, 'model.ckpt'))
        ########## Evaluation 
        net.eval()
        with torch.no_grad():
            current_test_data = test_data.sample(frac=1).reset_index(drop=True)
            running_loss_test = 0.0
            loss_epoch_test = 0
            loss_epoch_count_test = 0
            iteration_test = 0
            for start,end in zip(
                    range(0, len(current_test_data), batch_size),
                    range(batch_size, len(current_test_data)+1, batch_size)):
                iteration_test +=1
                current_batch = current_test_data[start:end]
                current_names = current_batch['Video_ID'].values
                current_videos=[]
                for i in range(batch_size):
                    data = np.load(video_sub_path + current_names[i] + '.npy' , allow_pickle = True)
                    current_video = data.reshape(1,num_frames,dim_image,dim_image)
                    current_videos.append(current_video)
                current_videos = np.array(current_videos)
                current_labels = current_batch['label'].values
                current_videos = torch.from_numpy(current_videos).float()
                current_labels = torch.from_numpy(current_labels).long()
                inputs, labels = current_videos.to(device) , current_labels.to(device)
                # forward 
                outputs = net(inputs)
                loss_test = criterion(outputs, labels)
                running_loss_test += loss_test.item()
                if iteration_test % 3 == 0:   
                    print('[%d, %5d] Test loss: %.3f' %
                        (epoch + 1, iteration_test, running_loss_test / 3))
                    loss_epoch_test += running_loss_test / 3
                    loss_epoch_count_test += 1 
                    running_loss_test = 0.0

            if np.mod(epoch + 1, 1) == 0:
                current_epoch_loss = loss_epoch_test / loss_epoch_count_test
                test_loss.append(current_epoch_loss)
        
    # Train accuracy
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        current_train_data = train_data.sample(frac=1).reset_index(drop=True)
        iteration = 0
        # print len(current_train_data)
        for start,end in zip(
                range(0, len(current_train_data), batch_size),
                range(batch_size, len(current_train_data)+1, batch_size)):
            iteration +=1
            current_batch = current_train_data[start:end]
            current_names = current_batch['Video_ID'].values
            current_videos=[]
            for i in range(batch_size):
                data = np.load(video_sub_path + current_names[i] + '.npy' , allow_pickle = True)
                current_video = data.reshape(1,num_frames,dim_image,dim_image)

                current_videos.append(current_video)
            current_videos = np.array(current_videos)

            current_labels = current_batch['label'].values

            current_videos = torch.from_numpy(current_videos).float()
            current_labels = torch.from_numpy(current_labels).long()

            inputs, labels = current_videos.to(device) , current_labels.to(device)
            # breakpoint()

            # forward 
            outputs = net(inputs)

            # Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]
            # Total number of labels
            total += len(labels)
            correct += (predicted == labels).sum()
        accuracy = 100 * correct / float(total)
        print('Accuracy of the network on the train data: %0.2f %%' % accuracy)

    # Test accuracy
    net.eval()
    videoid_list = []
    predict_list = []
    actual_list = []
    correct = 0
    total = 0
    with torch.no_grad():
 
        current_test_data = test_data.sample(frac=1).reset_index(drop=True)
        iteration = 0
        # print len(current_train_data)
        for start,end in zip(
                range(0, len(current_test_data), batch_size),
                range(batch_size, len(current_test_data)+1, batch_size)):
            
            iteration +=1
            current_batch = current_test_data[start:end]
            current_names = current_batch['Video_ID'].values
            current_videos=[]
            for i in range(batch_size):
                data = np.load(video_sub_path + current_names[i] + '.npy' , allow_pickle = True)
                current_video = data.reshape(1,num_frames,dim_image,dim_image)
                current_videos.append(current_video)
            current_videos = np.array(current_videos)
            current_labels = current_batch['label'].values
            current_videos = torch.from_numpy(current_videos).float()
            current_labels = torch.from_numpy(current_labels).long()
            inputs, labels = current_videos.to(device) , current_labels.to(device)
            # forward 
            outputs = net.forward(inputs , True)
            # Get predictions from the maximum value
            predicted = torch.max(outputs.data, 1)[1]
            videoid_list.append(current_names)
            predict_list.append(predicted.cpu().numpy())
            actual_list.append(labels.cpu().numpy())
            # Total number of labels
            total += len(labels)
            correct += (predicted == labels).sum()
        accuracy = 100 * correct / float(total)
        print('Accuracy of the network on the test data: %0.2f %%' % accuracy)

        df = pd.DataFrame({
            'Video_ID': videoid_list,
            'Prediction_Result': predict_list,
            'Accurate_Result': actual_list
            })
        df.to_csv(save_path,
            index=True,
            encoding='utf-8',
            columns=['Video_ID', 'Prediction_Result', 'Accurate_Result']
            )

    ## PLOT
    fig = plt.figure(figsize = [4,3], dpi = 600)
    width = 1.5
    ax = gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(width)
        
    tick_width = 0.5
    plt.tick_params(direction = 'in', width = tick_width)
        
    epochs = np.arange(n_epochs)
    plt.plot(epochs , train_loss, label = "Train")
    plt.plot(epochs , test_loss, label = "Test")
    plt.legend(loc='upper right', fontsize = 8, frameon = False)
    plt.ylabel("Loss", fontsize=8)
    plt.xlabel("Epochs", fontsize=8)
    plt.show()
    # plt.savefig('loss.png', bbox_inches='tight')

if __name__ == "__main__":
    main()