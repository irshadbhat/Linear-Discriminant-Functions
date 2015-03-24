# Linear-Discriminant-Functions

Implementation of different variants of Linear Perceptron

## Example:

    >>> from perceptron import Perceptron
    >>> 
    >>> X = [(1, 6), (7, 2), (8, 9), (9, 9), (4, 8), (8, 5), (2, 1), (3, 3), (2, 4), (7, 1), (1, 3), (5, 2)]
    >>> y = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    >>> 
    >>> testset = [(1, 5.5), (8, 1), (2, 6), (2, 4.5), (6, 1.5), (4, 3)]
    >>> 
    >>> # Single Sample Perceptron without Margin    
    ... clf = Perceptron(learning_rate=0.1, margin=0.0)
    >>> clf.MISSP(X, y)
    >>> clf.predict(testset)
    [1, 2, 1, 2, 2, 2]
    >>> clf.plot_boundary()
    ![alt tag](https://github.com/irshadbhat/Linear-Discriminant-Functions/SSP.png)
    >>> 
    >>>
    >>> # Single Sample Perceptron with Margin    
    ... clf = Perceptron(learning_rate=0.1, margin=1.0)
    >>> clf.MISSP(X, y)
    >>> clf.predict(testset)
    [1, 1, 1, 2, 2, 2]
    >>> clf.plot_boundary()
    ![alt tag](https://github.com/irshadbhat/Linear-Discriminant-Functions/MISSP.png)
    >>>
    >>> 
    >>> # Margin Infused Relaxation Algorithm
    ... clf = Perceptron(learning_rate = 2.0, margin=1.0)
    >>> clf.MIRA(X, y)
    >>> clf.predict(testset)
    [1, 1, 1, 2, 2, 2]
    >>> clf.plot_boundary()
    ![alt tag](https://github.com/irshadbhat/Linear-Discriminant-Functions/MIRA.png)


## Contact:

    Irshad Ahmad Bhat
    MS-CSE IIITH, Hyderabad
    bhatirshad127@gmail.com
    irshad.bhat@research.iiit.ac.in

