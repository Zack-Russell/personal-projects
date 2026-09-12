'''
Naive Bayes - Supervised learning lecture from Module 4 

Based on Bayes Theorum 

Features are independent of each other (this is the naive assumption) 
    which is likely untrue in practical application
All features have an equal effect on the result
'''
from pathlib import Path
from textblob import TextBlob 
from textblob.sentiments import NaiveBayesAnalyzer

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Starting with a very very simple approach grabbed from textblob, will expound further below
text = "Today is a beautiful day. Tomorrow looks like bad weather."
ml_blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())

# This returns a classification and percentages based on probability of positive and negative sentiment
print(ml_blob.sentiment)

for sentence in ml_blob.sentences:
    print(sentence.sentiment)


#There ends what school taught me about this thing. not quite enough for me. 
#Below is a classifier implementation from scratch using only numpy & core python

''' 
Gaussian naive bayes notes

Gaussian distribution refers, of course, to normal distribution


=== Bayes Theorem - P(A|B) = (P(B|A)xP(A))/(P(B)) ===
    So likelihood of event A occurring given that B is true (posterior probability) 
    = ((Likelihood of B given A multiplied by the likeihood of event A occurring sans new evidence)
    Divided by probability of B under all possible outcomes)

I've always read about this in context of medicine. 
    This formula would weigh a test with a high accuracy (~90%) 
    against the chance of a very rare disease (~1% of population)
    to find the true risk of being sick when the test returns a positive result

    In this example, P(A) is 1%. There's a 1% chance you have the disease before testing
    P(NotA) is 99%. There's a 99% chance you don't have said disease
    P(B|A) is 90%. The chance of a positive test given that you HAVE the disease is 0.90
    P(B|NotA) is 10%. The chance of a positive test given you don't have the disease is 10%

    P(B) = [0.90*0.01]+[0.10*0.99] = .009+.099 = 0.108, 10.8%
        In total, a test can be positive because you really have it or because it's a false alarm,
        so the entire probability of B given all scenarios is 10.8%
            This means 10.8% of the entire population will test positive, even though 1% are sick!

    So P(A|B) = (.90*.01)/(.108) = 0.0833 ---- if you test positive, you have an 8.33% chance of having it fr

    
=== Naive bayes === 

In practice, we're looking for: 
"what is the probability that the data provided is some class k given x?"

So... P(y=k|x) = (P(x|y=k)*P(y=k))/P(x)

Gaussian naive bayes computes 2 probabilities and uses the greater one. We ignore the denominator here (P(x)).


Making our FINAL equation P(y=k|x) = (P(x|y=k)*P(y=k))


Data --> yes or no / cat or dog / tumor is malignant or benign / etc.

In reality, P(x|y=k) is very complex. The height and weight of a person are NOT independent, 
they are functional determinants
    So we legit just assume they're indepedent features lol

    P(x|y=k) is equal to product(j through number of features where j init = 1) of P(xj|y=k)

    We assume the probability of a feature occurring given that class is k is the same as 
    calculating the probability of this feature value for the mean and variance in a normal distribution 
        given the mean and variance of the respective class

Presume we have 100 observations, 30 of them are class k. 
    P(y=k) is 30% until proven otherwise, in a vacuum (this is the first half of equation)
    P(xj|y=k) is (1/sqrt(2 * pi * variance ^2 for all kj)) * e^-((xj-mean of all kj)^2/2(variance^2 for all kj))
    

The last thing we really worry about is underflow. Don't want to multiply miniscule probabilities together.
    Log(P(y=k|x)) is proportional to (∝) Log(P|y=k) + (for j in range(num features): log(P(xj|y=k)))
    
Now how do u turn this into code? 
We functionally argmax k, finding maximum p of the logarthmic probabilities
'''

class GausianNB:


    def fit(self, X, y):
    # X is our data matrix, y is the correct classification
    # more specifically X is a 2d feature matrix and y is a 1 dimensional set of class labels 

        # turn these values into numpy arrays
        X, y = np.asarray(X), np.asarray(y)

        # get the number of unique values from y to know how many classes we can have
        self._classes = np.unique(y) 

        # number of classes is the length of unique values in array y
        # number of features is dimension 1 of array X where dim 0 is the number of observations
        # for our example dataset this is 30 features with 2 possible classes, 0 or 1
        n_classes, n_features = len(self._classes), X.shape[1]

        # for each class and each feature we want means, so we init some zerod matrices
        self._means = np.zeros((n_classes, n_features)) # 1 row per class, 1 column per indepdent var/feature

        '''
        visualizing _means would be like this if we had 3 features (independent vars) and 2 classes (k)
        where μ is the mean
            feature 0   feature 1   feature 2
    class 0   μ₀,₀        μ₀,₁         μ₀,₂
    class 1   μ₁,₀        μ₁,₁         μ₁,₂₉
        '''

        self._variances = np.zeros((n_classes, n_features)) # same shape as above, empty for now
        # the prior is how common the target class is overall
        self._priors = np.zeros(n_classes) # 1 dimensional array containing the number of classes only

        # index is which column we are currently filling in, k is total number of classes
        # this loops through our list of unique classes to add means to the indices we specify
        # so this leverages the fact that arrays/matrices are MUTABLE to perform these operations
        for index, k in enumerate(self._classes): 
            # new matrix housing all input data in matrix X where correct classification is k
            Xk = X[y==k] # all the values where y = our target class

            # the mean of this particular set of data per feature is put in row index of the targeted matrix for the targeted feature
            self._means[index] = Xk.mean(axis=0) # axis = 0 collapses the rows, averaging across observations with 1 mean per feature
            # the variances are treated the same
            self._variances[index] = Xk.var(axis=0)

            # number of instances where our target class is k divided by entire sample size
            self._priors[index] = Xk.shape[0] / X.shape[0] # yields fraction of total set that belongs to class k

        # This doesn't have to return anything, I think. The work happens up top to fit.
        # But this line does allow us to chain it together like .fit().predict()
        return self 


    def _log_gaussian(self, X):
        # Now it's confusing because the person teaching this made it all broadcasted arrays with no looping
        
        # This section satisfies the right hand side of -0.5(log(2π*var(kj)) - ((xj - mu kj)²)/2(var(kj))
            # this is expressed broadly as P(xⱼ | y=k) = 1/√(2π·var(kj)) · exp( -(xⱼ - μ(kj))² / (2·var(kj)) ) if we weren't taking log odds
                # but the logarithmic transformation makes this into subtraction, kills exponent
            # The final formula is born out of the math operations from working out the logs of (P(x|y=k)*P(y=k)), basically

        # X[:, None, :] is the numpy array evalulation where we get...
                    # All of the first dimension (samples/observations), no second dimension (classes), all third dimension (features)
                    # this is just done for broadcasting cleanliness against an (n_classes, n_features) matrix
        num = -0.5 * (X[:, None, :] - self._means)**2 / self._variances
        # presume we have 114 samples and 30 features with 2 classes
        # X[:, None, :] has shape (114, 1, 30)
        # self._means has shape (2,30), which broadcasts to (1,2,30)
        # at the end of the day this reads as: for every sample, class, and feature, compute (xⱼ - μ(kj))²
            # man this is confusing

        # bc both sides are negative, it doesn't matter which is subtracted from which here
        # now we do the other half of the equation, -0.5(log(2π*var(kj))
        # this is again broadcasting a (2,30) matrix to be 114, 2, 30
        # we just add -0.5·log(2π·var) to every existing term in num
        log_prob = num - 0.5 * np.log(2 * np.pi * self._variances)

        # final step is collapsing the array of (114,2,30) down to (114,2), that is every observation and every class
        # now we just add up the log-densities across features, getting us to log_likelihood[j,k] = P(x_i | y=k) for sample j and class k
        return log_prob.sum(axis=2)

    def predict(self, X):
        # make sure it's an array again
        X = np.asarray(X)

        # below we apply bayes rule in log form: log P(y=k|x) ∝ log P(x|y=k) + log P(y=k)
            # where ∝ means is proportional to bc I be forgetting taht
        # so our log likelihood for each observation from last method is log P(x|y=k)
            # this is the probability of x where y is class k

        # get the log gaussian for every sample in our dataset as defined above and store it as a matrix
        log_likelihood = self._log_gaussian(X)
        # remember the priors are a 1dim array representing the total likelihood of each class
        # this is log P(y=k), the probability that any given observation in the dataset is class k
        log_prior = np.log(self._priors)

        # now we just have to find the maximum of the log likelihoods plus the prob that it's k in a vacuum
        return self._classes[np.argmax(log_likelihood + log_prior, axis =1)]


# return_x_y just skips recieving a bunch of unnecessary data from sk, returning a plain tuple instead
# where x is the data, y is the target
# so x is a 2d feature matrix and y is a 1d array of class labels
X, y = load_breast_cancer(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# call our fit method on a new instance of the class
classifier = GausianNB().fit(X=X_train, y=y_train)

# predictions on the test data are done
y_pred = classifier.predict(X_test)

# now we find an accuracy metric for the prediction value given the correct test data
print(f"Accuracy score for this seed: {accuracy_score(y_pred, y_test)}")

