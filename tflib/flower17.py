
from os import listdir
import numpy as np
import scipy.misc
import time
import pdb
Label={'01_Sunflower':0,
       '02_Daisy':1,
       '03_Iris':2,
       '04_Daffodil':3,
       '05_Pansy':4,
       '06_Bluebell':5,
       '07_Buttercup':6,
       '08_Colts Foot':7,
       '09_Cowslip':8,
       '10_Crocus':9,
       '11_Dandelion':10,
       '12_Fritillary':11,
       '13_Lily_Valley':12,
       '14_Snowdrop':13,
       '15_Tigerlily':14,
       '16_Tulip':15,
       '17_Windflower':16,     
       }

def make_generator(path, classes, batch_size, image_size, pharse='train'):
    epoch_count = [1]
    image_list = []
    for sub_class in classes:
        sub_class_path = path + '/'+ pharse + '/'+ sub_class
        sub_class_image = listdir(sub_class_path)        
        image_list.extend([sub_class_path + '/' + i for i in sub_class_image])

    def get_epoch():
        images = np.zeros((batch_size, 3, 64, 64), dtype='float32')
        labels = np.zeros((batch_size,), dtype='int32')
        files = range(len(image_list))
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        epoch_count[0] += 1
        for n, i in enumerate(files):
            #image = scipy.misc.imread("{}/{}.png".format(path, str(i+1).zfill(len(str(n_files)))))
            image = scipy.misc.imread("{}".format(image_list[i]))
            label = Label[image_list[i].split('/')[3]]
            image = scipy.misc.imresize(image,(image_size,image_size))
            images[n % batch_size] = image.transpose(2,0,1) / 255.
            labels[n % batch_size] = label
            if n > 0 and n % batch_size == 0:
                yield (images,labels)    
    
    return get_epoch

def load(batch_size, classes, data_dir='/datatmp/dataset/flower_10_100000_old',image_size = 64):
    return (
        make_generator(data_dir, classes, batch_size, image_size, pharse='train'),
        make_generator(data_dir, classes, batch_size, image_size, pharse='valid')
    )

if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0])
        if i == 1000:
            break
        t0 = time.time()

