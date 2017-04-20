#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """
    
    cleaned_data = []

    ### your code goes here
    for i in range(len(ages)):
        cleaned_data.append((ages[i][0],net_worths[i][0],abs(predictions[i][0]- net_worths[i][0])))
    result = sorted(cleaned_data,key = lambda x:x[2])[0:int(round(len(ages) * 0.9))]
    return result

