from imageai.Prediction import ImagePrediction
import os
execution_path=os.getcwd()

# instantiate
prediction = ImagePrediction()

# set model type, 
prediction.setModelTypeAsSqueezeNet()
prediction.setModelPath(os.path.join(execution_path, "squeezenet_weights_tf_dim_ordering_tf_kernels.h5"))
prediction.loadModel()

predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "godzilla.jpg"), result_count=5 )
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)

    #testing git2 

 a = "xz"