'''
IML 2018 Set 5, exercise 3

In order to run this, copy code from here:
https://mattpetersen.github.io/load-mnist-with-numpy
and save it to a file called 'mnist.py'.

'''
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mnist import mnist

def kmeans(X,k,init_centroids = None, max_iter = 200):
    '''
    Inputs
    X : ndarray
        Data matrix of size (n,d), rows are observations.
    
    k : integer
        Number of clusters.
        
    init_centroids : None or ndarray
        Initial centroids. If provided, this needs to be an array of size (k,d).
        
    max_iter : integer
        Maximum number of iterations (if not converged before).  
    
    Returns
    a : ndarray 
        A vector of length n, containing the cluster assignments for each data point.
        
    C : ndarray
        A matrix of size (k,d) containing the found cluster centroids as rows.
    '''
    n,d = X.shape
    
    # Initialization. The vector 'a' will hold the cluster assingments and 
    # the matrix 'C' the cluster centroids.
    
    # if no initial centroids are given, choose a random assignment
    if init_centroids is None:
        C = np.empty((k,d))
        a = np.random.choice(k,n)    
    # initial centroids given, now initial assignment is undefined (= None) 
    else:
        C = np.copy(init_centroids)
        a = None
        
    # main loop    
    for i in range(max_iter):
                
        # a) find clusters, update cluster means 
        if a is not None:        # (we skip this in the first iteration IF initial centroids were provided)
            for j in range(k):
                C[j,:] = np.mean(X[a == j,:],axis = 0) 
         
        # b) compute the distances and assign points to the nearest cluster
        dists = cdist(X,C, metric = "sqeuclidean")
        a_new = np.argmin(dists,axis = 1)
       
        # check for convergence
        if np.array_equal(a,a_new):
            print("Converged after %d iterations" % (i + 1))
            break
        
        a = np.copy(a_new)

    if i == (max_iter - 1):
        print("Max iterations reached")
        
    return a,C
        
def create_data_a(n):
    '''
    Sample data from a bivariate normal distribution 
    with zero-mean and identity covariance. 
    
    n : integer
        Number of samples.
    '''
    return np.random.normal(0,1,(n,2)) 
 
if __name__ == "__main__":
    ###########################################################################
    # 3a)   
    # Generate data and run k-means with k = 3 couple times. Plot the results.
    np.random.seed(13452)
    n = 100
    X = create_data_a(n)
    
    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    
    for ax in axes:
        a,C = kmeans(X,3)
        ax.scatter(X[:,0],X[:,1], c = a)
        ax.scatter(C[:,0],C[:,1], c = "r", marker="x", s = 4**3) 
    
    fig.suptitle("Running k-means with different initializations.")
    ###########################################################################        
    # 3b)
    # Load mnist data. The file mnist.py includes this code: 
    # https://mattpetersen.github.io/load-mnist-with-numpy    
    where_to_store_mnist = "./data/mnist"
    x_train, y_train, _ , _ = mnist(path = where_to_store_mnist)
    
    # class labels are one-hot encoded, 
    # transform one-hot vectors to single integers.
    y_train = np.nonzero(y_train)[1]
    
    # plot 5th training instance and its class
    inx = 4
    fig,ax = plt.subplots()
    ax.imshow(x_train[inx,:].reshape(28,28),cmap = "Greys")
    plt.title("class = %d" % y_train[inx])
    ###########################################################################        
    # 3c)
    
    # discard all but 500 first examples
    N = 500
    x_train = x_train[:N,:]
    y_train = y_train[:N]
    k = 10
    
    # run kmeans
    a,C = kmeans(x_train, k = k, init_centroids = x_train[:k,:])
    
    # number of examples to show
    n_examples = 5
    
    fig, axes = plt.subplots(k,n_examples + 1)
    
    for i in range(k):
        some_examples_inx = np.where(a == i)[0][:n_examples]
        
        for j in range(n_examples + 1):    
            ax = axes[i,j]
          
            if j == 0:
                img = C[i,:].reshape((28,28))
            else:
                img = x_train[some_examples_inx[j - 1],:].reshape((28,28))
                
            ax.imshow(img,cmap = "Greys")
            ax.set_xticks([])
            ax.set_yticks([])
                
    fig.suptitle("First 10 data points as initial cluster means.")
    ###########################################################################      
    # 3d)
    
    # select initial centroids
    init_C = np.zeros((k,x_train.shape[1]))

    for jj in range(k):
        init_C[jj,:] = x_train[ np.where(y_train == jj)[0][0],:]

    # run kmeans with different initilization
    a,C = kmeans(x_train, k = k, init_centroids = init_C)
    
    # number of examples to show
    n_examples = 6
    
    fig, axes = plt.subplots(k,n_examples + 1)
    
    for i in range(k):
        some_examples_inx = np.where(a == i)[0][:n_examples]
        
        for j in range(n_examples + 1):    
            ax = axes[i,j]
          
            if j == 0:
                img = C[i,:].reshape((28,28))
            else:
                img = x_train[some_examples_inx[j - 1],:].reshape((28,28))
                
            ax.imshow(img,cmap = "Greys")
            ax.set_xticks([])
            ax.set_yticks([])
                
    fig.suptitle("Initial cluster means selected according to class labels.")
    plt.show()