# dnn_xray
This code pack is a novel application of Neural Networks for X-Ray CT. In X-Ray CT, the objective is to find the
linear attenuation coefficients in image space from transmission data. The common approaches for this problem are:
- One-shot algorithms: Given that the source-detector geometry is "well-defined" and ignoring the randomness and noise components of data,
we can find the image coefficients by one-shot algorithms like Filtered Backprojection (FBP) and such.
- Iterative algorithms: We assume a model (Poisson or Weighted Gaussian) and solve the problem iteratively.

In both of these approaches, the system matrix H that defines the connection between image and data space is known and is used.

In our approach, the questions we are looking for answers are:
- Given that we have a vast amount of image-data pairs for a given fixed geometry, can we learn a highly non-linear inverse relationship
between data and image space?
- If we learn so, can we do better than the state-of-the-art methods in terms of speed/image quality?

Therefore, we first generated simulated data using MATLAB's radon transform and assumed forward projected estimate of the truth image
that govern Beer's law follows Poisson distribution and generated data this way. After generating dataset and splitting it, without
giving any information about the system, we trained our neural network having images as outputs and data as inputs. For this simulated
case, we were able to get a better mean performance than FBP method. These results will be posted in the blog in future.

File list:
- generate_data.m: Generate image & data pair in MATLAB.
- split_train_validation_test.m: Split this dataset into training, validation and test sets.
- train_xray_dnn.py: Train neural networks in Python using Theano and Keras libraries.
