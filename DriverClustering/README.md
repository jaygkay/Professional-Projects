### 1. KMEANS Clustering

Total Distances that the drivers have achieved is the one feature to decide if the driver is a safe driver.

His/Her driving score is another feature to decide it.

- Hypothesis: The Good driver's driving score would be still high even if the driver has driven for a long distance.

- Result: good as expected

- 4 centroids are selected to cluser the clients to Best/Good/Fair/Bad Drivers.


### 2. Safe Drivier

The 4 types of the driver cluster do not explain if a driver drives in safe or not.

To support a drivers safe driving, contextual, sensor, score data are selected to decide if a driver drives in safe.

- Guard Score, which explains about driver's safety based on the cars arounds, is a feature to be a target.

- SVM, RF, LR, and Gradient Boosting are utilized to be determined. 

- Result: the gps location can be extracted to support where the driver drives in dangers.
