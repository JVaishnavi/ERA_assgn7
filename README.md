#Comparing various models for MNIST dataset

## Model 1

### Target:

* Get the initial set up - training, testing, basic structure of the code, data loader etc.
* Keep the parameters as low as possible (~8000) to stay below the parameter limit for this assignment and see the initial training and test accuracy 

### Results:
* Parameters: 8204
* Best Training Accuracy: 60%
* Best Test Accuracy: 59.37%

### Analysis:
* Extremely bad accuracy
* Both train and test accracy are much below the target mark. Ways to improve it includes
    - Batch normalisation
    - Add GAP instead of 5x5 kernal
* Training accuracy is consistently more than the test accuracy - although not a lot, there are slight signs of over fitting. Test accuracy can be improved with respect to training accuracy by 
    - transforming training data
    - adding dropout layers

 ![image](https://github.com/JVaishnavi/ERA_assgn7/assets/11015405/c9b16c94-75ba-45ed-b996-9e886c1b382a)


## Model 2

### Target:

* Change in structure
    - Add batch normalisation
    - Remove second max pooling. In the initial run of model 2, I had 2 max pooling layers but the accuracy was detoriating as the pooling layer was close to the output layer. Hence removed it. 
    - Add GAP
    - Transform training data
    - Increased the parameters slighty to reach closer to target accuracy. (Will reduce that in the next run)
    - Added drop out layers

### Results:
* Parameters: 13968
* Best Training Accuracy: 98.76
* Best Test Accuracy: 99.44

### Analysis:
* Train accuracy is less than test accuracy (No over fitting) - This is due to adding train transformation and dropout layers
* Added batch normalisation, GAP to improve the accuracy
* Value is oscillating a lot (Learning rate issue)

### To do:

* Reduce the epoch to 15 for the final iterations
* To compensate for lesser epoch, reduce the batch size
* Accuracy fluctuating around a number. Add better learning rate formula
* Reduce number of parameters by changing the channel and kernal size.

![image](https://github.com/JVaishnavi/ERA_assgn7/assets/11015405/0a411aa5-74ae-438d-9f93-bb9d88492c08)


## Model 3

### Target:

* Add Exponential LR to reduce fluctuation in the accuracies
* Reduce the batch size to increase the number of times the weights are updated
* Reduce epoch to 15
* Reduce the number of channels to match the assignment requirements and still try to maintain the accuracy (Had multiple tries on this. (Sub-optimal channel sizes like 10, 20 are used in the code to stick to the model parameter requirements.)
* Higher difference between training and test means that the training accuracy has scope of increasing. Hence the skeleton is slightly changed to accomodate more layers. 

### Results:
* Parameters: 7986
* Best Training Accuracy: 98.48
* Best Test Accuracy: 99.41

### Analysis:
* Tried Exponential LR, step LR, LR on plateau. I got better accuracy on exponential LR. LR on plateau might yield better results had I used the right parameters. Open to suggestions on this one.
* The test accuracy touched 99.4 but not very consistent. Better LR formula might have given stable and consistent accuracy

![image](https://github.com/JVaishnavi/ERA_assgn7/assets/11015405/9c013da3-22ff-4f61-903d-c39b91810db6)
