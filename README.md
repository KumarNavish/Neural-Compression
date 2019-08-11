# Neural-Compression
* Deep convolutional and feed forward neural networks (CNNs) have recently achieved great success in many visual recognition tasks. However, existing deep neural network models are ***computationally intensive and memory intensive***, hindering their deployment
in devices with low memory resources or in applications with strict latency requirements. 

* Therefore, a natural thought is to perform model compression and acceleration in deep networks without significantly decreasing the model performance. 
* During the past few years, tremendous progress has been made in this area. In this work, we survey the recent advanced techniques for compacting and accelerating CNNs (and feed forward NN) model developed.
* These techniques are roughly categorized into four schemes:
  * parameter pruning and sharing, 
  * low-rank factorization, 
  * transferred/compact convolutional filters.
  
* We focus on both network compression plus its robustness against adversarial attacks.
