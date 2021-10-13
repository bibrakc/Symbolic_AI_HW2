# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import csv
import numpy as np
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    apartments = []
    rooms = []
    rent = []
    with open('hw2 rent prediction data.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(reader, None)  # skip the headers
        #print(spamreader)
        for row in reader:
            print(', '.join(row))
            #print(row[1 :])
            size_apart = row[3]
            bedrooms = size_apart[0]
            sittingrooms = size_apart[1]
            rooms += [[int(bedrooms), int(sittingrooms)]]
            rent += [int(row[5])]
            #new_row = row[1:3] + [bedrooms] + [bathrooms] + row[4:6]
            new_row = row[1:3] + row[4:5]
            #apartments += [row[1 :]]
            apartments += [new_row]

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
#print(apartments)
print("number of apartments = "+ str(len(apartments)))
num_apartments = len(apartments)
#print(rooms)
#print(rent)

all_distric = []
all_address = []
all_source = []

for distric, address, source in apartments:
    all_distric += [distric]
    all_source += [source]
    all_address += [address]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(all_distric)
numeric_district = le.transform(all_distric)

le.fit(all_address)
numeric_address = le.transform(all_address)


le.fit(all_source)
numeric_source = le.transform(all_source)

#print(numeric_district)
#print(numeric_address)
#print(numeric_source)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(rooms)

#print(scaler.data_max_)
scaled_rooms = scaler.transform(rooms)


# numeric_district + numeric_address + numeric_source + bedrooms + sittingrooms
transformed_appartments = []

for i in range(len(numeric_district)):
    transformed_appartments += [[numeric_district[i], numeric_address[i], numeric_source[i], scaled_rooms[i][0], scaled_rooms[i][1]]]

#print("Final Transformation !!!")
#print(transformed_appartments)
#print("Final Transformation !!! Leaving index 0 out")
#print(transformed_appartments[:0]+transformed_appartments[1:])


def customDistance(A, B):
    distric = 0
    address = 0
    source = 0

    if(A[0] != B[0]):
        distric = 1
    if (A[1] != B[1]):
        address = 1
    if (A[2] != B[2]):
        source = 1

    return np.sum( np.sqrt( distric+address+source + (A[3] - B[3])**2 + (A[4] - B[4])**2))


def customDistanceWeights(A, B):
    w = [0.9, 0.1, 0.4, 2, 0.7]
    distric = 0
    address = 0
    source = 0

    if(A[0] != B[0]):
        distric = 1
    if (A[1] != B[1]):
        address = 1
    if (A[2] != B[2]):
        source = 1

    return np.sum( np.sqrt( (w[0]*distric) + (w[1]*address) + (w[2]*source)
                    + (w[3]*(A[3] - B[3])**2) + (w[4]*(A[4] - B[4])**2)))


#print(customDistance(transformed_appartments[0], transformed_appartments[1]))

from sklearn.neighbors import KNeighborsRegressor
#neigh = KNeighborsRegressor(n_neighbors=2, metric=customDistanceWeights)
#neigh.fit(transformed_appartments[:8]+transformed_appartments[9:], rent[:8]+rent[9:])


my_KRegressors = [('k_neighbors_1', KNeighborsRegressor(n_neighbors=1, metric=customDistance)),
                  ('k_neighbors_2', KNeighborsRegressor(n_neighbors=2, metric=customDistance)),
                  ('k_neighbors_1_Weighted', KNeighborsRegressor(n_neighbors=1, metric=customDistanceWeights)),
                  ('k_neighbors_2_Weighted', KNeighborsRegressor(n_neighbors=2, metric=customDistanceWeights))]

error_average = {'k_neighbors_1': 0, 'k_neighbors_2': 0, 'k_neighbors_1_Weighted': 0, 'k_neighbors_2_Weighted': 0}
for name, regressor in my_KRegressors:
    for i in range(num_apartments):
        #print("performing regresion for "+ name + " as test case leaving out apartment " + str(i))
        regressor.fit(transformed_appartments[:i]+transformed_appartments[i+1:], rent[:i]+rent[i+1:])
        predict = regressor.predict([transformed_appartments[i]])
        error = np.fabs(predict - rent[i])
        error_percent = (error/rent[i])*100
        error_average[name] += error_percent
        #print(name + " predicts: " + str(predict) + " actual: "+ str(rent[i]) + " error: " + str(error) + " error%: "+ str(error_percent))
    error_average[name] = error_average[name]/(num_apartments-1)
    #print("Average Error for "+ name + ": " + str(error_average[name]))

print(error_average)
#print("Average Error for "+ name + ": " + str(error_average[name]))