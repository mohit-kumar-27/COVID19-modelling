import numpy as np
import pandas as pd 
from datetime import date
import datetime
import math
import matplotlib.pyplot as plt
import time

# preprocessing of data
original = pd.read_csv('../COVID19_data.csv')
data = original.copy()
data.columns = ['Date', 'Confirmed', 'Recovered', 'Deceased', 'Other', 'Tested', 'First Dose', 'Second Dose', 'Total Doses']
data.drop(['Deceased', 'Other', 'Recovered', 'Second Dose', 'Total Doses'], axis = 1, inplace = True)
data['Date'] = pd.to_datetime(data['Date'])
complete_data = data[data.Date >= '2021-03-08'].copy()
data = data[(data.Date >= '2021-03-08') & (data.Date <= '2021-04-26')].copy()
data = data.reset_index().drop('index', axis = 1)
data['Confirmed'] = data['Confirmed'].diff()
data['Tested'] = data['Tested'].diff()
data = data.drop(0, axis = 0).reset_index().drop('index', axis = 1)
data['Confirmed'] = data['Confirmed'].astype(int)
data['Tested'] = data['Tested'].astype(int)
data['First Dose'] = data['First Dose'].astype(int)

start = datetime.datetime(2021, 3, 16)
end = datetime.datetime(2021, 4, 26)
last_date = datetime.datetime(2021, 12, 31)
N = 70e6

# timeseries for average confirmed cases till September 20, 2021
gt = complete_data['Confirmed'].copy().to_numpy()
for i in range(len(gt) - 1, 7, -1):
    gt[i] = (gt[i] - gt[i-7]) / 7
gt = gt[8:]

# timeseries for average confirmed cases till April 26, 2021
average_confirmed = np.zeros(42)
ts = data.copy().to_numpy()
for day in range(42):
    for previous_day in range(7):
        average_confirmed[day] += ts[day + previous_day + 1][1] / 7 # average confirmed column till April 26, 2021

# timeseries for average first dose
first = data[['Date', 'First Dose']].copy().to_numpy()
for i in range(len(first) - 1, 6, -1):
    first[i][1] = (first[i][1] - first[i-7][1]) / 7
first = first[7:]

# extrapolate first dose data till December 31, 2021
while (last_date - first[-1][0]).days != 0:
    first = np.append(first, [[first[-1][0] + datetime.timedelta(days = 1), 200000]], axis = 0)

# timeseries for average tested till December 31, 2021
tested = complete_data[['Date', 'Tested']].copy().to_numpy()
for i in range(len(tested) - 1, 7, -1):
    tested[i][1] = (tested[i][1] - tested[i-7][1]) / 7
tested = tested[8:] # start timeseries from March 16, 2021

# extrapolate tested data till December 31, 2021
while (last_date - tested[-1][0]).days != 0:
    tested = np.append(tested, [[tested[-1][0] + datetime.timedelta(days = 1), tested[-1][1]]], axis = 0)

def generate_timeseries(beta, S0, E0, I0, R0, CIR0, V, tested, ndays, N, alpha, gamma, epsilon):
    S = np.zeros(ndays)
    E = np.zeros(ndays)
    I = np.zeros(ndays)
    R = np.zeros(ndays)
    e = np.zeros(ndays)

    S[0] = S0 
    E[0] = E0
    I[0] = I0 
    R[0] = R0

    start_date = datetime.datetime(2021, 3, 16)
    end_date = datetime.datetime(2021, 4, 26)
    waning_date = datetime.datetime(2021, 9, 11)
    last_date = datetime.datetime(2021, 12, 31)

    for day in range(ndays - 1):
        if day <= 30:
            deltaW = R0 / 30
        elif day >= 180:
            deltaW = R[day - 180] + epsilon * V[day - 180][1]
        else:
            deltaW = 0
        S[day + 1] = S[day] - beta * S[day] * I[day] / N - epsilon * V[day][1] + deltaW
        E[day + 1] = E[day] + beta * S[day] * I[day] / N - alpha * E[day]
        I[day + 1] = I[day] + alpha * E[day] - gamma * I[day]
        R[day + 1] = R[day] + gamma * I[day] + epsilon * V[day][1] - deltaW

    averageS = np.zeros(ndays)
    averageE = np.zeros(ndays)
    averageI = np.zeros(ndays)
    averageR = np.zeros(ndays)

    for day in range(ndays):
        average_count = 0
        for previous_day in range(day, day - 7, -1):
            if previous_day >= 0:
                averageS[day] += S[previous_day]
                averageE[day] += E[previous_day]
                averageI[day] += I[previous_day]
                averageR[day] += R[previous_day]
                average_count += 1
        averageS[day] = averageS[day] / average_count
        averageE[day] = averageE[day] / average_count
        averageI[day] = averageI[day] / average_count
        averageR[day] = averageR[day] / average_count

    for day in range(ndays):
        CIR = CIR0 * tested[0][1] / tested[day][1] 
        e[day] = averageE[day] / CIR 

    return averageS, averageE, averageI, averageR, e

def future_timeseries(BETA, S0, E0, I0, R0, CIR0, V, tested, ndays, N, alpha, gamma, epsilon, closed_loop = False):
    S = np.zeros(ndays)
    E = np.zeros(ndays)
    I = np.zeros(ndays)
    R = np.zeros(ndays)
    e = np.zeros(ndays)

    S[0] = S0 
    E[0] = E0
    I[0] = I0 
    R[0] = R0
    beta = BETA

    new_cases_every_day = []

    for day in range(ndays - 1):

        if closed_loop == True:
            if day % 7 == 1 and day >= 7:
                average_cases_last_week = 0
                for i in range(7):
                    CIR = CIR0 * tested[0][1] / tested[day - i][1] 
                    average_cases_last_week += alpha * (E[day - i]) / CIR 
                average_cases_last_week /= 7
                if average_cases_last_week < 10000:
                    beta = BETA
                elif average_cases_last_week < 25000:
                    beta = BETA * 2 / 3
                elif average_cases_last_week < 100000:
                    beta = BETA / 2 
                else:
                    beta = BETA / 3

        if day <= 30:
            deltaW = R0 / 30
        elif day >= 180:
            deltaW = R[day - 180] + epsilon * V[day - 180][1]
        else:
            deltaW = 0

        S[day + 1] = S[day] - beta * S[day] * I[day] / N - epsilon * V[day][1] + deltaW
        E[day + 1] = E[day] + beta * S[day] * I[day] / N - alpha * E[day]
        I[day + 1] = I[day] + alpha * E[day] - gamma * I[day]
        R[day + 1] = R[day] + gamma * I[day] + epsilon * V[day][1] - deltaW

        CIR = CIR0 * tested[0][1] / tested[day][1] 
        new_cases_every_day.append(alpha * E[day])
    
    averageS = np.zeros(ndays)
    averageE = np.zeros(ndays)
    averageI = np.zeros(ndays)
    averageR = np.zeros(ndays)

    for day in range(ndays):
        average_count = 0
        for previous_day in range(day, day - 7, -1):
            if previous_day >= 0:
                averageS[day] += S[previous_day]
                averageE[day] += E[previous_day]
                averageI[day] += I[previous_day]
                averageR[day] += R[previous_day]
                average_count += 1
        averageS[day] = averageS[day] / average_count
        averageE[day] = averageE[day] / average_count
        averageI[day] = averageI[day] / average_count
        averageR[day] = averageR[day] / average_count

    for day in range(ndays):
        CIR = CIR0 * tested[0][1] / tested[day][1] 
        e[day] = averageE[day] / CIR 

    return averageS, averageE, averageI, averageR, e, new_cases_every_day

def calculate_loss(beta, S0, E0, I0, R0, CIR0):
    _, _, _, _, e = generate_timeseries(beta, S0, E0, I0, R0, CIR0, first, tested, 42, 70000000, 1 / 5.8, 1 / 5, 0.66)
    ALPHA = 1 / 5.8 
    i = ALPHA * e 
    i_average = np.zeros(42)
    for t in range(42):
        count = 0
        for j in range(t, t - 7, -1):
            if j >= 0:
                count += 1
                i_average[t] += i[j]
            else:
                break 
        i_average[t] /= count 

    loss = 0
    for t in range(42):
            loss += (math.log(average_confirmed[t]) - math.log(i_average[t])) ** 2 
    loss /= 42
    return loss

def calculate_gradient(params):

    beta, E0, I0, R0, CIR0 = params[0], params[1], params[2], params[3], params[4]
    S0 = N - (params[1]+params[2]+params[3])

    f = calculate_loss(beta, S0, E0, I0, R0, CIR0)
    fR_beta = calculate_loss(beta + 0.01, S0, E0, I0, R0, CIR0)
    fR_E0 = calculate_loss(beta, S0, E0 + 1, I0, R0, CIR0)
    fR_I0 = calculate_loss(beta, S0, E0, I0 + 1, R0, CIR0)
    fR_R0 = calculate_loss(beta, S0, E0, I0, R0 + 1, CIR0)
    fR_CIR0 = calculate_loss(beta, S0, E0, I0, R0, CIR0 + 0.1)
    fL_beta = calculate_loss(beta - 0.01, S0, E0, I0, R0, CIR0)
    fL_E0 = calculate_loss(beta, S0, E0 - 1, I0, R0, CIR0)
    fL_I0 = calculate_loss(beta, S0, E0, I0 - 1, R0, CIR0)
    fL_R0 = calculate_loss(beta, S0, E0, I0, R0 - 1, CIR0)
    fL_CIR0 = calculate_loss(beta, S0, E0, I0, R0, CIR0 - 0.1)

    grad_beta = (fR_beta - fL_beta) / 0.02
    grad_E0 = (fR_E0 - fL_E0) / 2
    grad_I0 = (fR_I0 - fL_I0) / 2
    grad_R0 = (fR_R0 - fL_R0) /2
    grad_CIR0 = (fR_CIR0 - fL_CIR0) / 0.2
 
    grad = np.array([grad_beta, grad_E0, grad_I0, grad_R0, grad_CIR0])

    return grad, f

def project_params(params,bounds):
    '''projects parameters onto their natural boundary'''
    for i in range(0,5):
        if params[i] < bounds[i,0]:
            params[i] = bounds[i,0]
        elif params[i] > bounds[i,1]:
            params[i] = bounds[i,1]
    
    if params[1] + params[2] + params[3] > N:
        tot = params[1] + params[2] + params[3]
        params[1] = params[1] * (N/tot)
        params[2] = params[2] * (N/tot)
        params[3] = params[3] * (N/tot)
    
    return params

def gradient_descent(params, max_iterations = 10000):
    bounds = np.array([[0,1], [0,N], [0,N], [0.15*N, 0.36*N], [12,30]])
    i = 0
    while True:
        gradients, loss = calculate_gradient(params)
        if i % 1000 == 0:
            print('iteration no: {}, loss: {}'.format(i+1, loss))
        i += 1

        if i == max_iterations or loss < 0.01:
            print('iteration no: {}, loss: {}'.format(i+1, loss))
            break
        params = params - gradients * (1 / (i + 1))
        params = project_params(params, bounds)
        
    return params

def plot_new_cases_per_day(params, ndays = 188):
    beta = params[0]
    plt.figure(figsize = (9, 7))

    _, _, _, _, _, new_cases_every_day = future_timeseries(beta, params[1], params[2], params[3], params[4], params[5], first, tested, ndays, 70000000, 1/5.8, 1/5, 0.66, closed_loop=False)
    plt.plot(new_cases_every_day[42:], label = 'beta, open loop')

    _, _, _, _, _, new_cases_every_day = future_timeseries(beta * 2 / 3, params[1], params[2], params[3], params[4], params[5], first, tested, ndays, 70000000, 1/5.8, 1/5, 0.66, closed_loop=False)
    plt.plot(new_cases_every_day[42:], label = '2/3 beta, open loop')

    _, _, _, _, _, new_cases_every_day = future_timeseries(beta / 2, params[1], params[2], params[3], params[4], params[5], first, tested, ndays, 70000000, 1/5.8, 1/5, 0.66, closed_loop=False)
    plt.plot(new_cases_every_day[42:], label = '1/2 beta, open loop')

    _, _, _, _, _, new_cases_every_day = future_timeseries(beta / 3, params[1], params[2], params[3], params[4], params[5], first, tested, ndays, 70000000, 1/5.8, 1/5, 0.66, closed_loop=False)
    plt.plot(new_cases_every_day[42:], label = '1/3 beta, open loop')

    _, _, _, _, _, new_cases_every_day = future_timeseries(beta, params[1], params[2], params[3], params[4], params[5], first, tested, ndays, 70000000, 1/5.8, 1/5, 0.66, closed_loop=True)
    plt.plot(new_cases_every_day[42:], label = 'closed loop')

    plt.plot(gt, label = 'reported cases')

    plt.legend()
    
    if ndays == 188:
        end_date = 'September 20, 2021'
    else:
        end_date = 'December 31, 2021' 

    plt.title(f'open loop and closed loop predictions till {end_date}')
    plt.xlabel('days since April 26, 2021')
    plt.ylabel('No. of new cases every day')
    plt.show()

def plot_susceptible(params, ndays = 188):
    beta = params[0]
    plt.figure(figsize = (9, 7))

    S, _, _, _, _, _ = future_timeseries(beta, params[1], params[2], params[3], params[4], params[5], first, tested, ndays, 70000000, 1/5.8, 1/5, 0.66, closed_loop=False)
    plt.plot(S[42:], label = 'beta, open loop')

    S, _, _, _, _, _ = future_timeseries(beta * 2 / 3, params[1], params[2], params[3], params[4], params[5], first, tested, ndays, 70000000, 1/5.8, 1/5, 0.66, closed_loop=False)
    plt.plot(S[42:], label = '2/3 beta, open loop')

    S, _, _, _, _, _ = future_timeseries(beta / 2, params[1], params[2], params[3], params[4], params[5], first, tested, ndays, 70000000, 1/5.8, 1/5, 0.66, closed_loop=False)
    plt.plot(S[42:], label = '1/2 beta, open loop')

    S, _, _, _, _, _ = future_timeseries(beta / 3, params[1], params[2], params[3], params[4], params[5], first, tested, ndays, 70000000, 1/5.8, 1/5, 0.66, closed_loop=False)
    plt.plot(S[42:], label = '1/3 beta, open loop')

    S, _, _, _, _, _ = future_timeseries(beta, params[1], params[2], params[3], params[4], params[5], first, tested, ndays, 70000000, 1/5.8, 1/5, 0.66, closed_loop=True)
    plt.plot(S[42:], label = 'closed loop')

    plt.legend()

    if ndays == 188:
        end_date = 'September 20, 2021'
    else:
        end_date = 'December 31, 2021' 

    plt.title(f'open loop and closed loop predictions till {end_date}')
    plt.xlabel('days since April 26, 2021')
    plt.ylabel('No. of susceptible people')
    plt.show()

# "params" [beta, E0, I0, R0, CIR0] 
params = np.array([0.43557, 199999, 84999, 1200000, 29])
start = time.time()
params = gradient_descent(params)
S0 = N - (params[1]+params[2]+params[3]) 
loss = calculate_loss(params[0], S0, params[1], params[2], params[3], params[4])
print('loss = ', loss)
print('beta = ', params[0])
print('S0 = ', S0)
print('E0 = ', params[1])
print('I0 = ', params[2])
print('R0 = ', params[3])
print('CIR0 = ', params[4])

plot_new_cases_per_day(np.array([params[0], S0, params[1], params[2], params[3], params[4]]))
plot_susceptible(np.array([params[0], S0, params[1], params[2], params[3], params[4]]))
plot_new_cases_per_day(np.array([params[0], S0, params[1], params[2], params[3], params[4]]), 290)
plot_susceptible(np.array([params[0], S0, params[1], params[2], params[3], params[4]]), 290)
end = time.time()
print(f'Time Elapsed: {end-start:0.2f} secs')