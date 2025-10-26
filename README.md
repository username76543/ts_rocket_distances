# ts_rocket_distances
Hand-wavey explanation of the intuition and motivation for this transform.

Consider a well mixed, two class problem. Say you want to seperate the red points and the blue points.

<img width="800" height="800" alt="well_mixed" src="https://github.com/user-attachments/assets/a1e755e1-3780-40e4-9c2b-6fcb171ff591" />

Forming a classifier to do this will be difficult. It should be, I randomly generated the points.
But forming a classifier on this sample is trivial:

<img width="800" height="800" alt="poorly_mixed" src="https://github.com/user-attachments/assets/969547ac-c231-483d-b451-3e1f2197ad96" />

Top is one color, bottom is the other.

If you could find a way to consistently map the points in the first space to the second space such that the red points go on top and the blue points on bottom, you could easily create a classifier. As an example, say that x, y are training points of classes red and blue and z is an unknown point. You have a function f that seperates class 1 and 2 in the testing set and a distance function d.

Then you algorithm is

    def classify(z):
      if d(f(z), f(x)) < d(f(z), f(y)):
        return x.class
      else:
        return y.class


For most of the points in the poorly mixed example, this will return the correct classification. Accuracy will vary based on how far x and y are from the class boundary, but it is just a form of nearest neighbors. You are doing nearest neighbors after a function that seperates your datapoints better than the baseline dataset.

Of course, this provides no way of finding a function f. But if you don't know how to do something, you can always guess.

ROCKET is extremely good at generating transforms on time series. Most of the transforms are not terribly informative, which is why ridge regression regularizes their coefficients toward zero. But if you take a lot of guesses, you normally end up with a few that are good.

Distance functions on time series are also highly effective. But you can't use them as pooling operations for ROCKET, since you need at least two points for a distance to be meaningful. However, if a function cleanly seperates two classes, then randomly selecting points will likely pick points such that classes end up near each other.

These observations are the motivation behind the Convolutional Cartography Transform (ConCar). You do random convolutions like ROCKET, while hoping that your convolution cleanly seperates the classes of time series. For each convolution, you select a subset of your testing points as "points of interest". If your convolution produced well seperated neighborhoods of time series, points that are of that class will be near your "points of interest" and points that are not of that class will be far. Then you measure the distance of all of the testing points from your points of interest.

Most of the convolutions are not helpful, but if you use an algorithm that performs feature selection such as Ridge Regression or Random Forests, the few convolutions which did seperate your classes will have higher weight. You can now produce a more useful set of features for classification.

Does this work? Empirically, it sometimes improves on ROCKET with 1/20th of the kernels, but the distance calculations make it much slower. It is almost always better than other distance based approaches. It is worse than MiniRocket, MultiRocket, HYDRA and QUANT, but again, 1/20th of the kernel count. Adding more kernels should improve performance at the cost of linearly increasing time. I highly doubt I have saturated the kernel space and I still need to switch from the ROCKET kernels to MiniRocket's kernels.

But this approach has one additional useful property. As the size of your training set increases, there is a natural way to improve the quality of your features. Just add more points of interest. Right now, the points of interest scale logarithmically to the size of the dataset, which has been the best approach in my limited testing. But each datapoint slowly increases the quality of all other datapoints.

[Comparison Charts Go Here]

Some Observations:

From the convolution theorem, the ROCKET Convolutional kernels are equivalent to frequency space pointwise multiplication. So randomly selecting frequencies in frequency space to amplify or surpress is equivalent to convolutional mapping. Since ROCKET is so effective, we can consider the time domain distance function post convolution is equivalent to using a similarity measure where certain frequncies are amplified or surpressed. This process should only work in the cases where classes have class boundaries in the frequency domain. But the fact that it seems to work everywhere implies that frequency domain class clustering is common to almost all datasets.

As the number of samples increases, the importance of using a elastic distance function should decrease, since the probability of selecting a well aligned point of reference increases. I need to check where the shift occurs.
