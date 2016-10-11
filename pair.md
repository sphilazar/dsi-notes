## Profit Curves

In this exercise, we are going to calculate the expected value given our cost-benefit matrix for a variety of binary classifiers at different thresholds.

The data we'll be working with can be found in `data/churn.csv`.


### Cost Benefit and Profit Curve Example

Before we start coding, let's make sure we understand how to calculate the cost benefit matrix and profit curve. Take this really simple example of what the true labels are and the predicted probabilities.

Say these are our results:

| datapoint | true label | predicted probability |
| --------- | ---------- | --------------------- |
| 0 | 0 | 0.2 |
| 1 | 0 | 0.6 |
| 2 | 1 | 0.4 |

Here is our cost benefit matrix:

|                   | Actual Yes | Actual No |
| ----------------- | ---------- | --------- |
| **Predicted Yes** |          6 |        -3 |
| **Predicted No**  |          0 |         0 |

1. Write out the confusion matrix for each of the possible thresholds. There should be 4 confusion matrices.

2. Calculate the profit for each of these confusion matrices.


### Profit Curve Implementation

1. Clean up the churn dataset with pandas. You should be predicting the "Churn?" column. You can drop the "State", "Area Code" and "Phone" columns. Make sure to convert any yes/no columns to 1/0's.

2. Specify a cost-benefit matrix as a 2x2 `numpy` array. Each cell of the matrix will correspond to the corresponding cost/benefit of the outcome of a correct or incorrect classification. This matrix is domain specific, so choose something that makes sense for the churn problem. It should contain the cost of true positives, false positives, true negatives and false negatives in the following form:

    ```python
    [[tp   fp]
     [fn   tn]]
    ```

3. Since sklearn's [confusion matrix](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix) function creates a flip-flopped confusion matrix, you will appreciate having your own.

    Write a function `standard_confusion_matrix` which takes `y_true` and `y_predict` and returns the confusion matrix as a 2x2 numpy array of the form `[[tp fp] [fn tn]]`.

    You may use sklearn's `confusion_matrix` function here, but having it in the standard form will reduce the amount of banging your head against the wall.

    You can check how sklearn's function works by trying an example:
    
    ```python
    In [1]: from sklearn.metrics import confusion_matrix
    
    In [2]: y_true =    [1, 1, 1, 1, 1, 0, 0]
    
    In [3]: y_predict = [1, 1, 1, 1, 0, 0, 0]
    
    In [4]: confusion_matrix(y_true, y_predict)
    Out[4]:
    array([[2, 0],
           [1, 4]])
    ```

4. Write a function `profit_curve` that takes these arguments:
    * `cb`: your cost-benefit matrix
    * `predict_probas`: predicted probability for each datapoint (between 0 and 1)
    * `labels`: true labels for each data point (either 0 or 1)

    Here's the psuedocode for the `profit_curve` function. Note the similarity to building a ROC plot!

    ```
    function profit_curve(costbenefit_matrix, predict_probas, labels):
        Sort instances by their prediction strength (the probabilities)
        For every instance in decreasing order of probability:
            Set the threshold to be the probability
            Set everything above the threshold to the positive class
            Calculate the confusion matrix
            Compute the expected profit:
                - multiply each of the 4 entries in the confusion matrix by
                their associated entry in the cost-benefit matrix
                - sum up these values
                - divide by the total number of data points
        return a list of the profits
    ```

5. Test your `profit_curve` function on the same toy example from above:

    ```python
    probas = np.array([0.2, 0.6, 0.4])
    labels = np.array([0, 0, 1])
    cb = np.array([[6, -3], [0, 0]])
    ```

    Do you get the same results?

    **Note:** It's OK if you only have 3 results instead of 4 since your code doesn't do both extreme thresholds.

    You can graph it too, but it's very silly with such a small example:

    ```python
    profits = profit_curve(cb, probas, labels)
    percentages = np.arange(0, 100, 100. / len(profits))
    plt.plot(percentages, profits, label='toy data')
    plt.title("Profit Curve")
    plt.xlabel("Percentage of test instances (decreasing by score)")
    plt.ylabel("Profit")
    plt.legend(loc='lower right')
    plt.show()
    ```

6. Now you're ready to build the profit curve on the real data! You should implement the `plot_profit_curve` function which will take the following parameters:

    ```
    model, label, costbenefit, X_train, X_test, y_train, y_test
    ```

    I should be able to use it like this:

    ```python
    plot_profit_curve(LogisticRegression(),
                      'Logistic Regression',
                      np.array([[6, -3], [0, 0]]),
                      X_train, X_test, y_train, y_test)
    plt.legend(loc='lower right')
    plt.show()
    ```

7. Now use the following code snippet to compare several models.

    Note that if your `plot_profit_curve` function is calling `plt.show`, you will get multiple plots popping up instead of just one, so you should remove that.

    ```python
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.ensemble import RandomForestClassifier as RF
    from sklearn.ensemble import GradientBoostingClassifier as GBC
    from sklearn.svm import SVC

    cb = np.array([[6, -3], [0, 0]])
    models = [RF(), LR(), GBC(), SVC(probability=True)]
    for model in models:
        plot_profit_curve(model, model.__class__.__name__, cb,
                          X_train, X_test, y_train, y_test)
    plt.title("Profit Curves")
    plt.xlabel("Percentage of test instances (decreasing by score)")
    plt.ylabel("Profit")
    plt.legend(loc='lower right')
    plt.show()
    ```

8. What's the maximum profit that we can achieve and which model should we use to get it? What proportion of the customer base does this target?

## Sampling Methods

### Undersampling
Finish writing this function: 

```python
def undersample(X, y, target):
    """
    INPUT:
    X, y - your data
    target - the intended percentage of positive
             class observations in the output
    OUTPUT:
    X_undersampled, y_undersampled - undersampled data

    `undersample` randomly discards negative observations from
    X, y to achieve the target proportion
    """
    # determine how many negative (majority) observations to retain

    # randomly discard negative (majority) class observations

    return X_undersampled, y_undersampled
	
```

### Oversampling
Finish writing this function: 

```python
def oversample(X, y, target):
    """
    INPUT:
    X, y - your data
    target - the percentage of positive class 
             observations in the output
    OUTPUT:
    X_oversampled, y_oversampled - oversampled data

    `oversample` randomly replicates positive observations
    in X, y to achieve the target proportion
    """
    # determine how many new positive observations to generate

    # replicate randomly selected positive observations
    
    # combine new observations with original observations

    return X_oversampled, y_oversampled
	
```

### SMOTE (Synthetic Minority Oversampling Technique)  

SMOTE is a method to randomly generate new minority class observations according to this procedure:  

-  for each observation, find the k nearest neighbors  
-  randomly select an observation and one its k nearest neighbors, then generate a new observation with these properties:  
  -  it's in the minority class (i.e. ```y = 0```)
  -  the value of each feature is a random value between the value of that feature in the original observation and that of the nearest neighbor (i.e. ```x_original < x_new < x_neighbor``` for each feature and neighbor)

Finish writing this function to implement SMOTE.

```python
def smote(X, y, target, k=None):
    """
    INPUT:
    X, y - your data
    target - the percentage of positive class 
             observations in the output
    k - k in k nearest neighbors

    OUTPUT:
    X_oversampled, y_oversampled - oversampled data

    `smote` generates new observations from the positive (minority) class:
    For details, see: https://www.jair.org/media/953/live-953-2037-jair.pdf
    """
    # fit a KNN model

    # determine how many new positive observations to generate

    # generate synthetic observations
    
    # combine synthetic observations with original observations

    return X_smoted, y_smoted
```


## Comparing Methods
1.  Test your functions on the churn dataset with Logistic Regression.
2.  Try a range of target sample proportions with each of your sampling methods, and find the sample proportion that maximizes expected profit.
3. Try the test in #2 several times, with a new train/test split each time. Does the expected profit and optimal threshold vary substantially? How might you deal with this variance? 
