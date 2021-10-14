# HW 2
# Bibrak and Brain

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
le_district = preprocessing.LabelEncoder()
le_district.fit(all_distric)
numeric_district = le_district.transform(all_distric)
#print(le_district.classes_)
#print(le_district.transform(le_district.classes_))


le_address = preprocessing.LabelEncoder()
le_address.fit(all_address)
numeric_address = le_address.transform(all_address)

le_source = preprocessing.LabelEncoder()
le_source.fit(all_source)
numeric_source = le_source.transform(all_source)

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
    w = [0.2, 0.05, 0.2, 2, 0.7]
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

print("\n\nK Regression Part")
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
        #print("predicted rent = " + str(predict) + ", Test rent ", rent[i])
        #print(name + " predicts: " + str(predict) + " actual: "+ str(rent[i]) + " error: " + str(error) + " error%: "+ str(error_percent))
    error_average[name] = error_average[name]/(num_apartments-1)
    #print("Average Error for "+ name + ": " + str(error_average[name]))

print("Average Error for K Regressor: ")
print(error_average)
#print("Average Error for "+ name + ": " + str(error_average[name]))

print("\n\nK Neighbor Part and Case Base Adaption")
error_average = {'k_neighbors_1': 0, 'k_neighbors_2': 0, 'k_neighbors_1_Weighted': 0, 'k_neighbors_2_Weighted': 0}
error_max = {'k_neighbors_1': 0, 'k_neighbors_2': 0, 'k_neighbors_1_Weighted': 0, 'k_neighbors_2_Weighted': 0}
error_max_problem = {'k_neighbors_1': 0, 'k_neighbors_2': 0, 'k_neighbors_1_Weighted': 0, 'k_neighbors_2_Weighted': 0}
error_max_sol = {'k_neighbors_1': 0, 'k_neighbors_2': 0, 'k_neighbors_1_Weighted': 0, 'k_neighbors_2_Weighted': 0}



for name, regressor in my_KRegressors:
    for i in range(num_apartments):
        #print("\n\ni = ", i)
        leave_out_apartments = transformed_appartments[:i]+transformed_appartments[i+1:]
        #print(leave_out_apartments)
        leave_out_rent = rent[:i]+rent[i+1:]
        #print(leave_out_rent)
        regressor.fit(leave_out_apartments, leave_out_rent)
        num_k = 1
        if(name == "k_neighbors_2" or name == "k_neighbors_2_Weighted"):
            num_k = 2
        predict = regressor.kneighbors([transformed_appartments[i]], num_k, True)
        #print("For problem case: " , [transformed_appartments[i]] , "Predict: ", predict)
        predicted_case_1 = leave_out_apartments[predict[1][0][0]]
        predicted_rent_1 = leave_out_rent[predict[1][0][0]]



        if (name == "k_neighbors_2" or name == "k_neighbors_2_Weighted"):
            predicted_case_2 = leave_out_apartments[predict[1][0][1]]
            predicted_rent_2 = leave_out_rent[predict[1][0][1]]
            #predicted_rent += predicted_rent_2
            #pre


        #print("test case = ", [transformed_appartments[i]])
        #print("predicred case = ", predicted_case)
        #print("predicted rent = " + str(predicted_rent) + ", Test rent " , rent[i])

        test_district_encoded = transformed_appartments[i][0]
        #Yp_district_encoded = le_district.transform(["Yp"])
        #Rule 1
        if (test_district_encoded == le_district.transform(["Yp"]) and predicted_case_1[0] == le_district.transform(["Pd"])):
            print("They Match")
            predicted_rent_1 *=1.2

        if (name == "k_neighbors_2" or name == "k_neighbors_2_Weighted"):
            if (test_district_encoded == le_district.transform(["Yp"]) and predicted_case_2[0] == le_district.transform(["Pd"])):
                #print("They Match, Test = ")
                predicted_rent_2 *=1.2

        '''
        #Rule 2
        if (test_district_encoded == le_district.transform(["Yp"]) and predicted_case_1[0] == le_district.transform(["Pd"])):
            print("They Match")
            predicted_rent_1 *=1.2

        if (name == "k_neighbors_2" or name == "k_neighbors_2_Weighted"):
            if (test_district_encoded == le_district.transform(["Yp"]) and predicted_case_2[0] == le_district.transform(["Pd"])):
                #print("They Match, Test = ")
                predicted_rent_2 *=1.2
        '''


        predicted_rent = predicted_rent_1
        if (name == "k_neighbors_2" or name == "k_neighbors_2_Weighted"):
            predicted_rent += predicted_rent_2
            predicted_rent /= 2


        error = np.fabs(predicted_rent - rent[i])
        error_percent = (error/rent[i])*100
        error_average[name] += error_percent
        if(error_percent > error_max[name]):
            error_max[name] = error_percent
            error_max_problem[name] = [transformed_appartments[i]]
            error_max_sol[name] = [predicted_case_1]

    error_average[name] = error_average[name] / (num_apartments - 1)



print("Average Error for K Regressor Case Base: ")
print(error_average)
print(error_max)
print(error_max_problem)
print(error_max_sol)
#print("Average Error for "+ name + ": " + str(error_average[name]))