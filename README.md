# Symbolic_AI_HW2
Case based reasoning system



# Problem Set

This homework illustrates the fundamentals of case-based reasoning by coding and testing a simple case-based reasoning system.    Answers may be submitted as a well-commented .py file and a PDF file with (1) a sample of the I/O behavior of your program, and (2) responses to each written question, identified and numbered in red below.  Code should be in python 3.   Alternatively you may use a jupyter notebook.   Before submitting the notebook, please export to html to submit (unfortunately, apparently canvas has problems with .ipynb).

This assignment will be done in small groups (2-3).   Please read and follow the class pair programming guidelines.   You may discuss any questions with anyone in the class without acknowledgment but should not consult code from any source.

Your system will use CBR for a regression task, the prediction of apartment rents.   An excel file of data is online (data is from Pal and Shiu, Foundations of Soft Case-Based Reasoning, 2004).   The data was extracted from an apartment rental site and describes apartments by city district, address, type of apartment (encoded as two digits, the number of bedrooms and the number of sitting rooms), the source of the listing, and the price.   In the source column, "indi" stands for an individual landlord, and other codes stand for different realtors.   All the attributes except the price are called the "features" of the cases, and the flat representation is called a "feature vector" representation (what we're calling cases here are identical to the instances of instance based learning, but generally cases have richer/structured representations).

Part 0 (not to hand in):  Try out the Stanford Vision k-nn demo (Links to an external site.) to familiarize yourself with the behavior of k-nn (note that the demo is for classification rather than regression, but it illustrates the influence of instances).

Part 1:  Write a system to do instance-based regression to predict rents and perform leave-one-out testing to assess accuracy.   Your system will take as input the test data, a vector of feature weights (the values of the terms wi in the equation below), and a value k for the number of instances to consider.   Initially your system will do pure k-nearest neighbor (k-nn), averaging the predictions of the closest k cases. 

In order to identify the closest cases, your system will need to be able to calculate the distance between cases.  For this you will use Euclidean distance (the L2 norm) with feature weightings:

weighted euclidean distance.JPG

Using this will require you to develop procedures for calculating the distance between different types of features. For any two features, the distance function should result in distances in the interval [0, 1], with 0 for identical features and 1 for maximally distant values.  Distance functions must be return numeric values even for nominal features (e.g., the district name).  For nominal features its fine just to use the distance function d(x,y) = 0 if x = y, and 1 otherwise as a starting point, though you may also use a finer-grained function..

Q1:  Describe the method you use for calculating the feature distances for each feature.   

Do not worry about efficiency of case retrieval.

Part 2:  Test the system on the data, with unweighted features (i.e., all weights = 1), for k=1 and k=2, and Q2: report the average error.

Part 3:  Tune the system by defining unequal feature weights, based on your sense of feature importance, and again test the system on the data, with your new weights, for k=1 and k=2.  Q3: Report the weighting you chose, the rationale, and the average error.

Part 4:  Q4: State in English at least two plausible adaptation rules for this domain.  The rules should describe a possible difference between a retrieved case and a current problem, and the associated change to be made in the solution of the retrieved case.  For example, if the domain were predicting travel times, a rule might be: "If the old trip was not during rush hour and the new prediction is for rush hour, increase travel time by 20%"   

Part 5:  Add code to adapt the solutions of the retrieved cases before they are averaged, using your two rules.  The rules may be hard coded.

Part 6:  Test the revised system on the data, with your weighted features (i.e., all features are equally weighted), for k=1 and k=2, and Q5: report the average error.

Part 7:  Try tuning your weightings and/or adjusting your adaptation rules to improve performance.   Q6: How did you try adjusting them, why, and what is the best accuracy you can achieve?   The best-performing team will be asked to report to the class on their best weights and adaptation rules, and their process for finding them.

Part 8: Q7: Discuss your observations on feature weighting, the effect of the adaptation rules, and the performance you observed in the tests.  Please add any other observations that you would like here.

Q8: A statement of relative contributions of the group members.   Normally this will simply be "Equivalent", but please describe any special circumstances.

