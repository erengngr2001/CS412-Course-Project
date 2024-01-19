**OVERVIEW**
This project aims to predict student homework grades based on their ChatGPT conversations. All the chat histories belong to retrieved from GPT-3.5. You will find 2 scripts in this repository: cs412_project_mainfile.ipynb, and hw_score_predict_data_augment.ipynb. The first one is the main file you should use. The latter is used earlier in this project in order to examine the effect of data augmentation. It is recommended to check it out and try new ways yourself.

**METHODOLOGY**
1) Parsing:
The HTML parser was already provided for us for this project. However, some valuable information were separated. We reorganized the HTML parser code so that we have role, content, and create_time information kept within a single dictionary object. The keys of the dictionary are codes that correspond to a encrypted user, and the values are the role, content, create_time information corresponding to that user.

2) Prompt Matching and Preprocessing:
From the dictionary created above, we pulled the content which belong to users. They are identified with a "user" tag in their "role" field, and they are called "prompts". Then, we converted these prompts to a single list format. We also manually added the questions of the homework. After that, we preprocessed both prompts and questions. Preprocessing includes tokenization, converting to lowercase, removing punctiations, removing non-alphabetic tokens, stemming, and stopwword removal respectively. After preprocessing, we built tf-idf vectorizer upon those preprocessed prompts and questions. We also created a dictionary for that, so that it is easier to see the results with respect to a user code. Now, our dictionary is ready, so we calculated cosine similarity scores between questions and user prompts, and also built a dictionary for that.

3) Feature Engineering:
One way we tried to analyze the data is sentiment analysis. We calculated the polarity score of prompts, and put a tag based upon their compound score. Other than, we played around the time and prompt information. We calculated the number of user prompts, number of negative and positive labeled prompts, average characters per promt, total time and average time spent, and prompt similarity for each chat. We also included the number of some keywords such as error, no, thank, next, entropy as features.
As we have seen from histogram provided to us, there were some outliers in the data. To remove outliers, we only used the homeworks with a grade higher than 60.

5) Train/Test Split:
Our datasize is very small. Therefore, we tried different split values, but the best fitting for us appeared to be 80% train and 20% test. As a result, we got 83 training data and 21 test data.
Note1: In the previous parts, we calculated prompt similarity between questions and prompts. In this part, we also tried calculating prompt similarity between prompts of students and the prompts of students who get 100 from the homework by using jaccard similarity. This way, we slightly reduced MSE. However, due to object type conflicts, we decided to not use it for easing other operations.
Note2: We also tried data augmentation for increasing the dataset size. However, after several attempts, we decided that data augmentation is not a suitable method for such a dataset. Further analysis might be done for this part. Our attempts resulted in higher MSE scores, or overfitting.

4) Fitting a Model
For this part, we have tried several models. Each has its advantages and disadvantages. One should select the model that fits his/her needs. We decided that the most real-life usable model is a random forest with bagging. Here are our model trials:
   4.1) Linear Regression:
       It is very simple and easy to implement. However, we got 36 MSE.
   4.2) Decision Tree:
       After trying several values, we selected max_depth as 1, and we observed 18.6 MSE with this model. We also printed MSE values for each node (i.e. for 3 nodes since max_depth is 1), and exported its graph as pdf. In order to reduce the MSE, we tried bagging and boosting. However, after trying several hyperparameters and choosing the best fit, we got 2.1 MSE for train set, and 18.9 MSE for test set. Furthermore, we calculated R2 score for a deeper understanding of test MSE, and we got 0.15 R2 score for test set. We also tried cross-validation. But mean cross-validation MSE appeared to be 41.4. We also plotted some graphs. You can follow and see the results from there.
   4.3) Random Forest:
       What is better than a tree? A forest! With best witting parameters n_estimators=36 and max_depth=5, we built a random forest. As a result, we got 9 MSE for train data, and 15.2 MSE for test data. Also, R2 score for test data is also increased to 0.32. We also plotted the tree structure for that, and exported it as a pdf. Also, we added a 3D graph that shows the correlation of the features user_prompts, positive_promts, and negative_prompts.
       In order to develop this model, we decided to implement bagging upon it. As a result, we got 14.8 MSE for train data, and 13 MSE for test data. Although we seem to be a bit in underfit region, this seems to be the best model we got from this project. We also plotted some graphs for predictions. Also, we plotted a chart that shows the MSE difference of random forest model with and without bagging.
   4.4) SVM:
       The data is very sparse and multidimensional. Therefore, linear kernel did terrible. After trying different kernels, best fit appeared to be poly with C=2. Even so, we got 41.4 MSE for train data and 21.7 MSE for test data, which is even worse than our decision tree. We plotted some graphs for showing the data and residuals, and a 3D graph for observing correlation.
   4.5) Neural Network:
       Although we predicted that the dataset size is too small for building an NN model, we still wanted to try and see the result. However, the results were beyond terrible. This is by far the worst model we have built. We tried with a simple structure that has two layers, the first having relu activation and the second having linear activation. We compiled it with adamoptimizer and loss function is based upon MSE score. We tried 10 epochs with batch size 16. However, even the first epoch results in a terribly high loss value (loss: 19625019392.0000 - val_loss: 42865127424.0000). Therefore, although we tried another NN models with different parameters, we did not do a detailed analysis on it.

**RESULTS**
Some of the above mentioned results can be found here.
