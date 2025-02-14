# ConversationalChatbot

Designed and implemented a chatbot using a Feedforward Neural Network (FNN) for intent classification. The model tokenizes user queries, converts them into embedding vectors, and processes them through a neural network with two hidden layers (64 neurons each) using ReLU activation and dropout 20% to prevent overfitting. The final layer outputs 58 logits, which are converted into probabilities via softmax to determine the most probable intent. Trained using the Adam optimizer, the model achieved a training accuracy of \textbf{95.6\%} and a validation accuracy of 88.5%.
