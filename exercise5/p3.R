#data - should be a matrix of n (=rows) p(=columns)-dimensional vectors
#k - number of clusters
#maxIters - maximum number of iterations until the algorithm stops, if no convergence before that. Default=100

kMeans <- function(data, k, maxIters = 100) {
    n <- nrow(data) #number of data points
    p <- ncol(data) #dimensionality of the points
    centroids <- matrix(numeric(k*p), nrow = k) #matrix of the cluster centroids
    #for each data point vector, randomly allocate a cluster from k options
    clusters <- sample(k, n, replace = T)
    #now the first element of clusters is the cluster of the first data point (row) of the input data, etc.
    
    for(iter in 1:maxIters) {
        
    }
}

sample(10, 100, replace = T)
