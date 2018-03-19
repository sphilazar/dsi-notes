## Part 1: Profit Curves

In this exercise, we are going to calculate the expected value given our cost-benefit matrix for a variety of binary classifiers at different thresholds.

The data we'll be working with can be found in `data/churn.csv`.

### Cost/Benefit and Profit Curve Example

Before we start coding, let's make sure we understand how to calculate the points on a profit curve from a cost-benefit matrix. Take this really simple example of what the true labels are and the predicted probabilities.

Say we ran a model and got the following results:

| Observation | True Label | Predicted Probability |
|:-----------:|:----------:|:---------------------:|
|       0     |      0     |          0.2          |
|       1     |      0     |          0.6          |
|       2     |      1     |          0.4          |

Here is our cost benefit matrix:

|                        | Actual Positive | Actual Negative |
| ---------------------- |:---------------:|:---------------:|
| **Predicted Positive** |        6        |        -3       |
| **Predicted Negative** |        0        |         0       |

1. Write out the confusion matrix for each of the possible thresholds. There should be 4 confusion matrices.
2. Calculate the expected profit for each of these confusion matrices.

### Profit Curve Implementation

1. Clean up the churn dataset with pandas. You should be predicting the "Churn?" column. You can drop the "State", "Area Code" and "Phone" columns. Make sure to convert any yes/no columns to 1/0's.
2. Specify a cost-benefit matrix as a 2x2 numpy array. Each cell of the matrix will correspond to the corresponding cost/benefit of the outcome of a correct or incorrect classification. This matrix is domain specific, so choose something that makes sense for the churn problem. It should contain the benefit of true positives, false positives, true negatives and false negatives in the following form:

    ```python
    [[tp, fp]
     [fn, tn]]
    ```

3. Since sklearn's [confusion matrix](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix) function creates a flip-flopped confusion matrix, you will appreciate having your own that is ordered in the way you're familiar with.

    Write a function `standard_confusion_matrix()` which takes `y_true` and `y_predict` and returns the confusion matrix as a 2x2 numpy array of the form `[[tp, fp], [fn, tn]]`.

    You may use sklearn's `confusion_matrix()` function here, but having it in the standard form will reduce the amount of banging your head against the wall.

    You can check how sklearn's function works by trying an example:
    
    ```python
    In [1]: from sklearn.metrics import confusion_matrix
    
    In [2]: y_true = [1, 1, 1, 1, 1, 0, 0]
    
    In [3]: y_predict = [1, 1, 1, 1, 0, 0, 0]
    
    In [4]: confusion_matrix(y_true, y_predict)
    Out[4]:
    array([[2, 0],
           [1, 4]])
    ```

4. Write a function, `profit_curve()`, that takes these arguments:
  * `cost_benefit`: your cost-benefit matrix
  * `predicted_probs`: predicted probability for each datapoint (between 0 and 1)
  * `labels`: true labels for each data point (either 0 or 1)

    Here's the psuedocode for the `profit_curve()` function. Note the similarity to building a ROC plot!

    ```
    function profit_curve(cost_benefit, predicted_probs, labels):
        1. Sort instances by their prediction strength (the probabilities)
            - Add 1 at the beginning so that you consider all thresholds
        2. For every instance in decreasing order of probability:
            1. Set the threshold to be the probability
            2. Set everything above the threshold to the positive class
            3. Calculate the confusion matrix
            4. Compute the expected profit:
                - multiply each of the 4 entries in the confusion matrix by
                  their associated entry in the cost-benefit matrix
                - sum up these values
                - divide by the total number of data points
        3. Return an array of the profits and their associated thresholds
    ```

5. Test your `profit_curve()` function on the same toy example from above:

    ```python
    test_probs = np.array([0.2, 0.6, 0.4])
    test_labels = np.array([0, 0, 1])
    test_cost_benefit = np.array([[6, -3], [0, 0]])
    ```

    Do you get the same results?

    You can graph it too, but it's very silly with such a small example:

    ```python
    profits = profit_curve(test_cost_benefit, test_probs, test_labels)
    percentages = np.arange(0, 100, 100. / len(profits))
    plt.plot(percentages, profits, label='toy data')
    plt.title("Profit Curve")
    plt.xlabel("Percentage of test instances (decreasing by score)")
    plt.ylabel("Profit")
    plt.legend(loc='best')
    plt.show()
    ```

6. Now you're ready to build the profit curve on the real data! Build off the plotting code above to implement a `plot_profit_curve()` function which will take the following parameters:

    ```
    model, cost_benefit, X_train, X_test, y_train, y_test
    ```

    You should be able to use it like this:

    ```python
    plot_profit_curve(LogisticRegression(),
                      np.array([[6, -3], [0, 0]]),
                      X_train, X_test, y_train, y_test)
    plt.legend(loc='best')
    plt.show()
    ```

    Note, if you have a sklearn model stored as a variable, say `model`, you can use `model.__class__.__name__` to find out what the model is named.

7. Now use the following code snippet to compare several models.

    Note, if your `plot_profit_curve()` function is calling `plt.show()`, you will get multiple plots popping up instead of just one, so you should remove that.

    ```python
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.ensemble import RandomForestClassifier as RF
    from sklearn.ensemble import GradientBoostingClassifier as GBC
    from sklearn.svm import SVC

    cost_benefit = np.array([[6, -3], [0, 0]])
    models = [RF(), LR(), GBC(), SVC(probability=True)]
    for model in models:
        plot_profit_curve(model, cost_benefit, X_train, X_test, y_train, y_test)
    plt.title("Profit Curves")
    plt.xlabel("Percentage of test instances (decreasing by score)")
    plt.ylabel("Profit")
    plt.legend(loc='best')
    plt.show()
    ```

8. What's the maximum profit that we can achieve, at what threshold and which model should we use to get it? What proportion of the customer base does this target?

## Part 2: Sampling Methods

Frequently we will to classify on datasets that do not have each class proportionally represented. You're aware of three main ways to counteract some of the negative effects class imbalance can cause on your models. Now you need to implement them below.

Throughout the rest of this part you will need to fill out a sampling method function according to its doc string. Some notes/pseudocode on/for coding are included for guidance. In addition, take a look at the [`numpy.random`](https://docs.scipy.org/doc/numpy/reference/routines.random.html) module, it will be very useful of the randomization tasks you'll need to perform in these functions.

### Undersampling

```python
def undersample(X, y, tp):
    """Randomly discards negative observations from X & y to achieve the
    target proportion of positive to negative observations.

    Parameters
    ----------
    X  : ndarray - 2D
    y  : ndarray - 1D
    tp : float - range [0, 1], target proportion of positive class observations

    Returns
    -------
    X_undersampled : ndarray - 2D
    y_undersampled : ndarray - 1D
    """
    # determine how many negative (majority) observations to retain in order to
    # attain target proportion

    # randomly discard that many negative (majority) class observations

    # return remaining, undersampled, observations

    return X_undersampled, y_undersampled
```

### Oversampling

```python
def oversample(X, y, tp):
    """Randomly choose positive observations from X & y, with replacement
    to achieve the target proportion of positive to negative observations.

    Parameters
    ----------
    X  : ndarray - 2D
    y  : ndarray - 1D
    tp : float - range [0, 1], target proportion of positive class observations

    Returns
    -------
    X_undersampled : ndarray - 2D
    y_undersampled : ndarray - 1D
    """
    # determine how many new positive observations to generate to attain in
    # order to attain target proportion

    # randomly select required number of positive observations with replacement
    
    # combine new observations with original observations

    # return new, oversampled, observations

    return X_oversampled, y_oversampled
```

### SMOTE - Synthetic Minority Oversampling TEchnique

SMOTE is a method to randomly generate new minority class observations according to following procedure:  

1. For each observation in the minority class, find the k nearest neighbors.
2. Randomly select both an observation and one of its k nearest neighbors.
3. Generate a new synthetic observation from these two points via:  
    1. Find the vector between the observation and its neighbor.
    2. Add randomness to this vector by walking a random percentage along each of its dimensions.
    3. Add this new vector to the original observation to make the synthetic observation.
    4. Label this synthetic observation as a member of the minority class (i.e. `y = 1`)

```python
def smote(X, y, tp, k=None):
    """Generates new observations from the positive (minority) class.
    For details, see: https://www.jair.org/media/953/live-953-2037-jair.pdf

    Parameters
    ----------
    X  : ndarray - 2D
    y  : ndarray - 1D
    tp : float - [0, 1], target proportion of positive class observations

    Returns
    -------
    X_smoted : ndarray - 2D
    y_smoted : ndarray - 1D
    """
    # fit a KNN model, why not use sklearn's?

    # determine how many new positive observations to generate

    # generate synthetic observations
    
    # combine synthetic observations with original observations

    return X_smoted, y_smoted
```

## Comparing Methods

1. Test your functions on the churn dataset with Logistic Regression.
2. Try a range of target sample proportions with each of your sampling methods, and find the sample proportion that maximizes expected profit.
3. Try the test in #2 several times, with a new train/test split each time. Does the expected profit and optimal sampling proportion vary substantially? How might you deal with this variance? 

## Extra Credit

Take a look at the code that you wrote in Part 1. Are there any unnecessary computations taking place? How can we reduce the amount of computations?
