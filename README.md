# Evaluating Multilayer Perceptron Models with Different Activation Functions
# Introduction
In this assignment, we explore the performance of multilayer perceptron (MLP) models using three different activation functions: ReLU, LeakyReLU, and Sigmoid. Activation functions play a crucial role in the performance of neural networks by introducing non-linearity, which allows the model to learn complex patterns. Our objective is to compare these activation functions in terms of training accuracy, validation accuracy, and overall model performance.
# Model Setup
We built and trained three MLP models on a standard dataset (Digits dataset from scikit-learn) using the following activation functions:
1.	ReLU (Rectified Linear Unit)
2.	LeakyReLU
3.	Sigmoid
Each model consisted of two hidden layers, each with 64 neurons, and was trained for 20 epochs. The final layer used a softmax activation function to classify the input into one of ten classes.
# Results
The performance of each model is visualized in the graphs below, which show the training and validation accuracy over the course of 20 epochs.
# •	ReLU Model:
* Training Accuracy: The ReLU model quickly achieved high training accuracy, approaching 1.0 within a few epochs.
* Validation Accuracy: The validation accuracy also improved rapidly, stabilizing around 0.95. The gap between training and validation accuracy indicates slight overfitting, but the model generalizes well to unseen data.
# •	LeakyReLU Model:
*	Training Accuracy: Similar to the ReLU model, the LeakyReLU model achieved high training accuracy, also close to 1.0.
*	Validation Accuracy: The validation accuracy was comparable to the ReLU model, stabilizing around 0.95. The training curve shows a more consistent increase in accuracy, possibly due to the LeakyReLU's ability to handle negative inputs better than ReLU.
# •	Sigmoid Model:
*	Training Accuracy: The Sigmoid model started slower than the other two, reflecting the sigmoid function's tendency to suffer from the vanishing gradient problem, especially in deeper networks.
*	Validation Accuracy: Despite the slower start, the Sigmoid model eventually reached a validation accuracy close to 0.92, though it lagged slightly behind the ReLU and LeakyReLU models.
# Interpretation
•	ReLU vs. LeakyReLU: Both ReLU and LeakyReLU showed strong performance, with minimal differences in accuracy. ReLU is a popular choice due to its simplicity and efficiency, but LeakyReLU can provide better performance in scenarios where neurons might otherwise "die" due to the ReLU's zero gradient for negative inputs.
•	Sigmoid: The Sigmoid activation function performed adequately but was outpaced by ReLU and LeakyReLU. The slower convergence and slightly lower accuracy suggest that Sigmoid might not be the best choice for deeper networks, as it is more prone to the vanishing gradient problem.
•	Overall Accuracy: All models demonstrated the capability to generalize well, with the ReLU and LeakyReLU models performing slightly better than the Sigmoid model. The slight overfitting observed in the ReLU and LeakyReLU models is a common occurrence in neural networks, especially when using highly flexible activation functions like ReLU.
 
 
# Conclusion
This assignment highlights the importance of choosing the right activation function based on the specific characteristics of the problem and the network architecture. While ReLU and LeakyReLU are generally more effective for deep learning tasks, Sigmoid can still be useful in certain contexts, particularly for smaller or simpler networks. The key takeaway is that while ReLU and its variants are preferred for their efficiency and performance, understanding the nuances of each activation function allows for more informed decisions in model design.



