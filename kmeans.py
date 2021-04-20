
import numpy as np
import cv2
   
def digits():        
    image = cv2.imread('data/digits.png')
    gray_img = cv2.cvtColor(image,
                            cv2.COLOR_BGR2GRAY)
    divisions = list(np.hsplit(i,100) for i in np.vsplit(gray_img,50))
    NP_array = np.array(divisions)
    train_data = NP_array[:,:50].reshape(-1,400).astype(np.float32)
    test_data = NP_array[:,50:100].reshape(-1,400).astype(np.float32)
    k = np.arange(10)
    train_labels = np.repeat(k,250)[:,np.newaxis]
    test_labels = np.repeat(k,250)[:,np.newaxis]
    knn = cv2.ml.KNearest_create()
    knn.train(train_data,
            cv2.ml.ROW_SAMPLE, 
            train_labels)
    
    # obtain the output from the
    # classifier by specifying the
    # number of neighbors.
    ret, output ,neighbours,distance = knn.findNearest(test_data, k = 3)
    
    # Check the performance and
    # accuracy of the classifier.
    # Compare the output with test_labels
    # to find out how many are wrong.
    matched = output==test_labels
    correct_OP = np.count_nonzero(matched)
    
    #Calculate the accuracy.
    accuracy = (correct_OP*100.0)/(output.size)
    3   
    # Display accuracy.
    print(accuracy)

def chars():
    # Load the data and convert the letters to numbers
    data= np.loadtxt('data/letter-recognition.data', dtype= 'float32', delimiter = ',',
                        converters= {0: lambda ch: ord(ch)-ord('A')})
    # Split the dataset in two, with 10000 samples each for training and test sets
    train, test = np.vsplit(data,2)
    # Split trainData and testData into features and responses
    responses, trainData = np.hsplit(train,[1])
    labels, testData = np.hsplit(test,[1])
    # Initiate the kNN, classify, measure accuracy
    knn = cv2.ml.KNearest_create()
    knn.train(trainData, cv2.ml.ROW_SAMPLE, responses)
    ret, result, neighbours, dist = knn.findNearest(testData, k=5)
    correct = np.count_nonzero(result == labels)
    accuracy = correct*100.0/10000
    print( accuracy )
chars()