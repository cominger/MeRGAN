
from os import listdir
import os
import numpy as np
import cPickle as pickle
import scipy.misc
import time
import pdb
Label={'airplane':0,
       'automobile':1,
       'bird':2,
       'cat':3,
       'deer':4,
       'dog':5,
       'frog':6,
       'horse':7,
       'ship':8,
       'truck':9}
train_data=[]
train_targets=[]
test_data=[]
test_targets=[]
def pre_processing(path):
    train_list = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
    test_list = ['test_batch']

    global train_data, train_targets
    global test_data, test_targets
  
    #load numpy image -train
    for file_name in train_list:
      file_path = os.path.join(path,file_name)
      with open(file_path, 'rb') as f:
        entry = pickle.load(f)#, encoding='latin1')
        train_data.append(entry['data'])
        if 'labels' in entry:
          train_targets.extend(entry['labels'])
        else:
          train_targets.extend(entry['fine_labels'])

    train_data = np.vstack(train_data).reshape(-1,3,32,32)
    train_data = train_data.transpose((0,2,3,1))
    train_targets = np.asarray(train_targets)

    #load numpy image - test
    for file_name in test_list:
      file_path = os.path.join(path,file_name)
      with open(file_path, 'rb') as f:
        entry = pickle.load(f)#, encoding='latin1')
        test_data.append(entry['data'])
        if 'labels' in entry:
          test_targets.extend(entry['labels'])
        else:
          test_targets.extend(entry['fine_labels'])

    test_data = np.vstack(test_data).reshape(-1,3,32,32)
    test_data = test_data.transpose((0,2,3,1))
    test_targets = np.asarray(test_targets)

def make_generator(path, classes, batch_size, image_size, pharse='train'):
    epoch_count = [1]
    image_list = []
    target_list = []
    data = targets = []

    global train_data, train_targets
    global test_data, test_targets

    if pharse == 'train':
      data = train_data
      targets = train_targets
    else:
      data = test_data
      targets = test_targets

    for sub_class in classes:
        sub_class_target = Label[sub_class]
        data_index = [targets == sub_class_target]
        image_list.extend(data[data_index])
        target_list.extend(targets[data_index])
  
    def get_epoch():
        images = np.zeros((batch_size, 3, 32, 32), dtype='float32')
        labels = np.zeros((batch_size,), dtype='int32')
        files = range(len(image_list))
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        epoch_count[0] += 1
        for n, i in enumerate(files):
            image = image_list[i]
            label = target_list[i]
            image = scipy.misc.imresize(image,(image_size,image_size))
            images[n % batch_size] = image.transpose(2,0,1) / 255.
            labels[n % batch_size] = label
            if n > 0 and n % batch_size == 0:
                yield (images,labels)    
    
    return get_epoch

def load(batch_size, classes, data_dir='/datatmp/dataset/CIFAR_10_100000_old',image_size = 32):
    if len(train_data) < 1:
      pre_processing(data_dir)

    return (
        make_generator(data_dir, classes, batch_size, image_size, pharse='train'),
        make_generator(data_dir, classes, batch_size, image_size, pharse='val')
    )

if __name__ == '__main__':
    pdb.set_trace()
    #total ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    train_gen, valid_gen = load(64,['airplane','automobile'], 'dataset/CIFAR10',32)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        if i == 1000:
            break
        t0 = time.time()

