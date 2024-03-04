import numpy as np


def get_xy_sort(length,num_ints=3):
    """
    Returns a pair of input and output for a sorting problem
    of sorting a sequence of integers of length "length" with
    integers from 0 to "num_ints".

    Example:

    length = 5
    num_ints = 4

    seq = [0,4,2,3,3]
    sol = [0,2,3,3,4]
    
    then we get

    x = cat(seq,sol[:-1]) =[0,4,2,3,3 ,0,2,3,3] 
    y = cat(seq[1:],sol) = [4,2,3,3 ,2,3,3,4]


    This is based on the code from the minGPT project by Karpathy et al.
    found in https://github.com/karpathy/minGPT/blob/master/demo.ipynb

    """

    seq = np.random.randint(0,num_ints, size=(length,))
    sol = np.sort(seq)
    cat = np.concatenate((seq, sol), axis=0)
    x = cat[:-1]
    y = cat[1:]

    return x,y


def get_train_test_sorting(length, num_ints, samples_per_batch,n_batches_train, n_batches_test):
    """
    Generates a dataset for sorting (see docstring of get_xy_sort() for details)
    The dataset is split into a training and test set.

    Returns a dictionary data with keys 'x_train', 'y_train', 'x_test', 'y_test' with the following shapes:
        - x_train: (n_batches_train,samples_per_batch, 2*length-1)
        - y_train: (n_batches_train,samples_per_batch, 2*length-1)
        - x_test: (n_batches_test,samples_per_batch, length)
        - y_test: (n_batches_test,samples_per_batch, length)
    
    """


    x_train = np.zeros((n_batches_train,samples_per_batch, 2*length-1))
    y_train = np.zeros_like(x_train)
    x_test = np.zeros((n_batches_test,samples_per_batch, 2*length-1))
    y_test = np.zeros_like(x_test)


    for j in range(samples_per_batch):
        for i in range(n_batches_train):
            x_train[i,j],y_train[i,j] = get_xy_sort(length,num_ints)
        for i in range(n_batches_test):
            x_test[i,j],y_test[i,j] = get_xy_sort(length,num_ints)

    data = {}
    data['x_train'] = x_train
    data['y_train'] = y_train

    #we only select the unsorted seq from x_test
    data['x_test'] = x_test[:,:,:length]

    #and we select the sorted seq from y_test
    data['y_test'] = y_test[:,:,length-1:]
    return data


def get_train_test_addition(n_digit,samples_per_batch = 1000,n_batches_train = 3, n_batches_test = 1):

    """
    Generates a dataset for addition (a + b = c) of n_digit numbers.
    The dataset is split into a training and test set.

    Note! The order of the of the digits in c is reversed.
    This is done, since it (for some reason) makes the addition easier to learn for the.

    Returns a dictionary data with keys 'x_train', 'y_train', 'x_test', 'y_test' with the following shapes:
            - x_train: (n_batches_train,samples_per_batch,n_digit*3)
            - y_train: (n_batches_train,samples_per_batch,n_digit*3)
            - x_test: (n_batches_test,samples_per_batch,n_digit*2)
            - y_test: (n_batches_test,samples_per_batch,n_digit+1)

    Example:
    n_digit = 2
    a = [3,4]
    b = [4,1]
    c = [0,7,5]
    
    then we get

    x = cat(a,b,c_reversed[:-1]) = [3,4, 4,1, 5,7]
    y = cat(a[1:],b,c_reversed) =  [4, 4,1, 5,7,0]


    This is based on the code from the minGPT project by Karpathy et al.
    found in https://github.com/karpathy/minGPT/tree/master/projects/adder
    """
    
    # total number of possible addition problems with ndigit numbers
    num = (10**n_digit)**2

    #return error if we don't have enough samples to fill the batches
    if (n_batches_train+n_batches_test)*samples_per_batch > num:
        return ValueError('Not enough samples for training and testing')

    #shuffle the indices
    perm = np.random.permutation(num)
    nd = 10**n_digit

    def get_xy(n_digit,idx):
        idx = perm[idx]
        a = idx // nd
        b = idx % nd

        c = a + b
        # encode the digits of a, b, c into strings
        astr = f'%0{n_digit}d' % a
        bstr = f'%0{n_digit}d' % b
        cstr = (f'%0{n_digit+1}d' % c)[::-1] # reverse c to make addition easier
        render = astr + bstr + cstr
        dix = [int(s) for s in render]

        x = dix[:-1]
        y = dix[1:]
        return x,y
    
    x_train = []
    y_train = []
    x_test= []
    y_test = []

    #collecting x,y pairs for training
    for i in perm[:n_batches_train*samples_per_batch]:
        x,y = get_xy(n_digit,i)
        x_train.append(x)
        y_train.append(y)

    #collecting x,y pairs for testing
    for i in perm[n_batches_train*samples_per_batch : (n_batches_train+n_batches_test)*samples_per_batch]:
        x,y = get_xy(n_digit,i)
        x_test.append(x)
        y_test.append(y)

    x_train = np.reshape(np.stack(x_train),(n_batches_train,samples_per_batch,n_digit*3))
    y_train = np.reshape(np.stack(y_train),(n_batches_train,samples_per_batch,n_digit*3))
    x_test = np.reshape(np.stack(x_test),(n_batches_test,samples_per_batch,n_digit*3))
    y_test = np.reshape(np.stack(y_test),(n_batches_test,samples_per_batch,n_digit*3))


    data = {}

    data['x_train'] = x_train
    data['y_train'] = y_train

    #we only select a and b from x_test
    data['x_test'] = x_test[:,:,:n_digit*2]
    #and we only select c from y_test (and reverse it)
    data['y_test'] = y_test[:,:,n_digit*2-1:][:,:,::-1]
    return data




def text_to_training_data(n_max, text_string,num_batches=64,batch_size=100):

    """
    Generates a dataset for training a transformer model of a text string.

    Input:
        - n_max: maximum length of sequence to be used in transformer model
        - text_string: string with the text to be used for training
        - num_batches: number of batches to be used for training
        - batch_size: number of sequences in each batch (each sequence has length n_max)
    
    Output:
        - data: dictionary with training data:
            - x_train: list of numpy arrays with input sequences (integers < m)
            - y_train: list of numpy arrays with output sequences 

        - idx_to_text_dict: dictionary to convert from index / integer to text
        - text_to_idx_dict: dictionary to convert from text to index / integer
        - m: size of vocabulary (number of unique characters in text_string)


    This is based on the code from the minGPT project by Karpathy et al.
    found in https://github.com/karpathy/minGPT
    
    """

    #Get unique characters in text_string
    chars = sorted(list(set(text_string)))
    data_size, m = len(text_string), len(chars)
    print('data has %d characters, %d unique.' % (data_size, m))

    #Create dictionaries to convert from text to index and viceversa
    text_to_idx_dict = { ch:i for i,ch in enumerate(chars) }
    idx_to_text_dict = { i:ch for i,ch in enumerate(chars) }


    chars_per_batch = n_max*batch_size
    batches_x = []
    batches_y = []

    #We split the string into num_batches 
    for i in range(num_batches):
        chunk = text_string[i*chars_per_batch: (i+1)*chars_per_batch + 1]
        idx = [text_to_idx_dict[s] for s in chunk]

        #The x are shifted by one character, lacking the last character
        batches_x.append(np.array(idx[:-1]).reshape(batch_size, n_max))

        #y lacks the first character,
        #such that x[i+1] = y[i] for i = 0,1,...,n_max-1.
        batches_y.append(np.array(idx[1:]).reshape(batch_size, n_max))

    
    data = {'x_train':batches_x,'y_train':batches_y}

    return data, idx_to_text_dict, text_to_idx_dict, m
