# ts_rocket_distances
Hand-wavey explanation of the intuition and motivation for this transform.

Consider a well mixed, two class problem. Say you want to seperate the red points and the blue points.

<img width="800" height="800" alt="well_mixed" src="https://github.com/user-attachments/assets/a1e755e1-3780-40e4-9c2b-6fcb171ff591" />

Forming a classifier to do this will be difficult. It should be, I randomly generated the points.
But forming a classifier on this sample is trivial:

<img width="800" height="800" alt="poorly_mixed" src="https://github.com/user-attachments/assets/969547ac-c231-483d-b451-3e1f2197ad96" />

Top is one color, bottom is the other.

If you could find a way to consistently map the points in the first space to the second space such that the red points go on top and the blue points on bottom, you could easily create a classifier. As an example, say that x, y are training points of classes red and blue and z is an unknown point. You have a function f that seperates class 1 and 2 in the testing set and a distance function d.

Then your algorithm is

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

<img width="1300" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/01d3c4a6-c28a-4ef9-b97a-b152fadcc922" />

First test of small datasets. Distance measure is MSM, 512 ROCKET kernels. All other classifiers use their default configurations from the Aeon toolkit. ConCar (labeled as poi due to weirdness in my code) had the best performance on several datasets, despite low kernel count. ROCKET ranks look deflated by inclusion of multiple forms of ROCKET.

Of course, this comparison isn't entirely fair. Let's try scaling up the number of kernels, but switching from MSM distance to euclidean distance. If the elasticity of the measure mattters more than the number of kernels, we should expect the relative performance of the ConCar transformation to decrease.

<img width="1300" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/2a23807e-d363-483b-aa4c-083c9d2c1aeb" />
Euclidean distances, 10000 convolutional kernels. It is clearly better than nn, not much else.

To further our apples to apples comparison, let's limit all the ROCKET methods to 840 kernels (since MiniRocket requires a multiple of 84) and see how they compare. We will have to exclude HYDRA and QUANT from this test, since they don't have as easy of a way to limit the algorithms.

<img width="1300" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/fae92977-af57-4350-9bbb-657e601f7fdc" />


So ConCar is a little worse than the default featureset, but still performs the best on several datasets if all algorithms have to use the same number of kernels. This also makes it clear that Ridge Regression definitely outperforms Random Forests for ConCar, which was unknown because of its hybrid distance and convolutional structure.

Some Observations:

From the convolution theorem, the ROCKET Convolutional kernels are equivalent to frequency space pointwise multiplication. So randomly selecting frequencies in frequency space to amplify or supress is equivalent to convolutional mapping. Since ROCKET is so effective, we can consider the time domain distance function post convolution is equivalent to using a similarity measure where certain frequncies are amplified or supressed. This process should only work in the cases where classes have class boundaries in the frequency domain. But the fact that it seems to work everywhere implies that frequency domain class clustering is common to almost all datasets.

As the number of samples increases, the importance of using a elastic distance function should decrease, since the probability of selecting a well aligned point of reference increases. I need to check where the shift occurs.

Note on Datasets:

All tests conducted on these datasets from the Aeon Time Series Classification repository (https://timeseriesclassification.com). The list is below, but was chosen from the following criteria
1. Series Length below 500 to keep the distance calculations down.
2. Excessively high dimensionality for the same reasons.
3. Series Length above 9, since ROCKET requires at least a series length of 8.
4. Excessive number of datapoints. Most of these are in MONSTER and I am just trying this idea out, not trying for top performance.

short_500_len_datasets = ['SmoothSubspace', 'MelbournePedestrian', 'ItalyPowerDemand', 'Chinatown', 'JapaneseVowels', 'RacketSports', 'LSST', 'Libras', 'FingerMovements', 'NATOPS', 'SharePriceIncrease', 'SyntheticControl', 'SonyAIBORobotSurface2', 'ERing', 'SonyAIBORobotSurface1', 'PhalangesOutlinesCorrect', 'ProximalPhalanxOutlineCorrect', 'MiddlePhalanxOutlineCorrect', 'DistalPhalanxOutlineCorrect', 'ProximalPhalanxTW', 'ProximalPhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxTW', 'DistalPhalanxTW', 'DistalPhalanxOutlineAgeGroup', 'TwoLeadECG', 'MoteStrain', 'SpokenArabicDigits', 'ElectricDevices', 'ECG200', 'MedicalImages', 'BasicMotions', 'TwoPatterns', 'CBF', 'SwedishLeaf', 'BME', 'EyesOpenShut', 'FacesUCR', 'FaceAll', 'ECGFiveDays', 'ECG5000', 'ArticularyWordRecognition', 'PowerCons', 'Plane', 'GunPointOldVersusYoung', 'GunPointMaleVersusFemale', 'GunPointAgeSpan', 'GunPoint', 'UMD', 'Wafer', 'Handwriting', 'ChlorineConcentration', 'Adiac', 'Epilepsy2', 'Colposcopy', 'Fungi', 'WalkingSittingStanding', 'Epilepsy', 'Wine', 'Strawberry', 'ArrowHead', 'ElectricDeviceDetection', 'WordSynonyms', 'FiftyWords', 'Trace', 'ToeSegmentation1', 'Coffee', 'DodgerLoopWeekend', 'DodgerLoopGame', 'DodgerLoopDay', 'CricketZ', 'CricketY', 'CricketX', 'FreezerRegularTrain', 'FreezerSmallTrain', 'UWaveGestureLibraryZ', 'UWaveGestureLibraryY', 'UWaveGestureLibraryX', 'UWaveGestureLibrary', 'Lightning7', 'ToeSegmentation2', 'DiatomSizeReduction', 'FaceFour', 'GestureMidAirD3', 'GestureMidAirD2', 'GestureMidAirD1', 'Symbols', 'HandMovementDirection', 'Heartbeat', 'Yoga', 'OSULeaf', 'Ham', 'Meat', 'Fish', 'Beef', 'FordA', 'FordB']

These short length datasets were excluded for taking to long on my machine. Most have too many samples or too many dimensions.
long_sub_500 = ['Tiselac', 'FaceDetection', 'PEMS-SF', 'Sleep', 'EMOPain', 'MindReading', 'MotionSenseHAR', 'PhonemeSpectra', 'DuckDuckGeese', 'Crop']
