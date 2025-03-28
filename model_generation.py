import numpy as np #Numpy
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics.pairwise import cosine_similarity
from hmmlearn import hmm
import pickle

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
telecom_data = pd.read_csv("./uom_project/telecom_customer_churn.csv")

#Now we only select the following Data Elements so as to make the data processing more faster and easier to Hot Encode.
telecom_data = telecom_data[["Phone Service","Internet Service","Contract", "Customer Status", "Churn Category"]]

#Now we proceed to Hot encode the data which we have derived from the csv file. 
telecom_model_data = pd.get_dummies(telecom_data, dtype=pd.Int32Dtype())

#We remove any possible "Nan: Data attribute that can interfere with our Data processing
telecom_model_data = telecom_model_data.ffill().dropna()

#Now we convert our data to a Numpy array (.to_numpy() function)
telecom_model_data_preprocess = telecom_model_data.to_numpy()


#Now we process the data to make it readable by the startprob_, transmat_ and emissionprob_ attributes of the model.
#We are inputting the data manually rather than deriving the samples from the model.
#This simple routine converts the data to a sum of 1.
#It is done by multiplying 1(Sum_of array) with each element of the array.

telecom_model_data_preprocess_norm = telecom_model_data_preprocess.sum(axis=1)
telecom_model_data_preprocess_norm = 1/telecom_model_data_preprocess_norm
row = 0
for x in telecom_model_data_preprocess_norm:
    telecom_model_data_preprocess[row] = np.dot(telecom_model_data_preprocess[row], x)
    row += 1
    
# Now we send the data to teh Hidden Markov Model Learn library. We process the data using Categorical Hidden Markov Model
# We assume the data is Discrete in nature, hence Categirical HMM would be the best suited for this problem
# We use the "viterbi" Algorithm in the HMM. I will go detail into the viterbi algorithm in the project report.

telecom_model = hmm.CategoricalHMM(n_components=len(telecom_model_data_preprocess[0]), algorithm="viterbi", init_params='se')
telecom_model.startprob_ = telecom_model_data_preprocess[0]
telecom_model.transmat_ = telecom_model_data_preprocess[:15]
telecom_model.emissionprob_ = telecom_model_data_preprocess

"""
To use the generated model .pkl file
-----------------------------------------------
with open("telecom_model.pkl", "rb") as file:
    telecom_model = pickle.load(file)
"""
#Now we get the Hot Encoded data converted to 32 bit integer    
X = telecom_model_data.astype(np.int32)

#We remove any negative values in our dataset and replace it with zero. 
X[X<0] = 0

#Now finally, we fit the data into our HMM_model
telecom_model.fit(X, len(X))

#Now we get the model to predict the possible probabilities of the states using the predict function of hmmlearn
samples = telecom_model.predict(X)

#We again remove the zeros in the samples
samples[samples<0] = 0


#We convert the prediction data into arrays contain 15 elements, similar to the original arrays
#This is particularly useful in converting it back to a Pandas DataFrame
#We use a simple routine to do the same.

samples_rows = (len(samples)//15)
samples = samples[:(samples_rows*15)] 
samples = np.reshape(samples, (samples_rows, 15))

#Now we convert the data to only 0 and 1, similar to teh original Dataset. This is to converts the probabilities into the corresponding states.
#We use the median as the cutoff value for the probabilities.

threshold = int(np.median(samples))
samples[samples<=threshold] = 0
samples[samples>threshold] = 1

#We convert it to a pandas DataFrame
sample_data = pd.DataFrame(samples)
sample_data.columns = telecom_model_data.columns

#We finally output the data to two seperate csv files, one is the original value, other is the generated one.
sample_data.to_csv("generated_data.csv")
telecom_model_data.to_csv("Data.csv")

#Finally the processing is Done
print("Processing Done!")


#Now we find the cosine similarities of both the original dataset and the generated data samples
cosine_similarity_array = cosine_similarity(telecom_model_data, sample_data)

#Now we print the avg value of all the similarity cosines
print(f"\n The Average cosine similarity of the two datasets is: {np.average(cosine_similarity_array)}")

with open("telecom_model.pkl", "wb") as file:
    pickle.dump(telecom_model, file)
    
    
