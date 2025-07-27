import numpy as np

np.set_printoptions(suppress=True, precision=2)

data = np.array([
    # User 0
    [[72, 5400, 6.5, 2100, 5.5],
     [75, 5200, 6.0, 2200, 6.0],
     [70, 4800, 6.2, 2000, 5.0],
     [74, 5100, 6.3, 2100, 5.8],
     [73, 5000, 6.4, 2150, 5.2],
     [76, 5300, 6.1, 2250, 6.5],
     [71, 4900, 6.5, 2050, 5.9]],
    
    # User 1
    [[80, 8000, 5.0, 2500, 7.5],
     [85, 8200, 5.1, 2600, 8.2],
     [78, 7900, 5.0, 2550, 7.8],
     [82, 8100, 5.2, 2650, 8.5],
     [84, 8300, 5.3, 2700, 8.1],
     [86, 8500, 5.1, 2750, 8.0],
     [81, 8000, 5.2, 2600, 7.9]],
    
    # User 2
    [[65, 3000, 8.0, 1800, 4.0],
     [68, 3100, 7.9, 1850, 3.8],
     [64, 2900, 8.1, 1750, 4.2],
     [67, 3200, 8.0, 1800, 4.1],
     [66, 3300, 8.2, 1850, 3.9],
     [69, 3400, 7.8, 1900, 4.0],
     [63, 2800, 8.3, 1750, 3.7]],
    
    # User 3
    [[90, 10000, 4.5, 3000, 9.5],
     [92, 10200, 4.2, 3100, 9.9],
     [88, 9800, 4.3, 3050, 9.8],
     [95, 10400, 4.0, 3200, 10.0],
     [91, 10100, 4.4, 3100, 9.6],
     [89, 9900, 4.1, 3000, 9.7],
     [94, 10300, 4.3, 3150, 9.9]]
])

# Shape: (4 users, 7 days, 5 features)
#Features: heart_rate, steps, sleep_hours, calories, screen_time

print(f"the shape of data: {data.shape}")
print(f"the total number of users: {data.shape[0]}")
print(f"the total number of days per user: {data.shape[1]}")
print(f"the total number of  features per day per user: {data.shape[2]}")
print(f"each user 1st 2 days data: {data[:,:2,:]}")

data[:,4:,1] += 750
data[:,:,4] *= 0.95

steps = data[:,:,1:2]
sleep_hours = data[:,:,2:3]
calories = data[:,:,3:4]
screen_time = data[:,:,4:5]

fitness_score = (steps/1000) + sleep_hours + (calories/1000) - screen_time
print(f"fitness scores: {fitness_score}")

steps_ = data[:,:,1]
sleep_hours_ = data[:,:,2]
calories_ = data[:,:,3]
screen_time_ = data[:,:,4]
heart_rate_ = data[:,:,0]

filtered = (steps_ > 9000) & (sleep_hours_ < 6)
print(f"users days  when  steps_ > 9000 and sleep_hours_ < 6: {data[filtered]}")

filtered_ = (screen_time_ > 8) & (heart_rate_ > 85)
print(f"users days when screen_time > 8 and heart_rate_ > 85: {data[filtered_]}")

high_sleep_hrs = sleep_hours_ > 7.5
high_sleep_hrs_  = np.any(high_sleep_hrs, axis=1)
print(f"user's days with high sleep hrs: {data[high_sleep_hrs_]}")

low_heart_rate = np.where(heart_rate_ < 65)
print(low_heart_rate)

data_ = np.nditer(data, flags=['multi_index'])
while not data_.finished:
    index = data_.multi_index
    value = data_[0] 
    print(f"index: {index}, value: {value}")
    data_.iternext()
    

data_ = np.nditer(data, flags=['multi_index'])
while not data_.finished:
    index = data_.multi_index
    value = data_[0] 
    if (index[2] == 2) & (value < 5):
        user = index[0]
        day = index[1]
        print(f"⚠️ Low Sleep – User: {user}, Day: {day}")
        print(data[user,day])
    data_.iternext()
    
data_copy = data.copy()
print(f"data before making change in copy: {data}")
data_copy[:,:,0] =70
print(f"data after making change in copy: {data}")

data_view = data.view()
print(f"data before making change in view: {data}")
data_view[:,:,3] = 9999
print(f"data after making change in view: {data}")

data__ = np.nditer(data, flags=['multi_index'],op_flags=['readwrite'])
while not data__.finished:
    index_ = data__.multi_index
    value_ = data__[0]
    sleep = data[index_[0], index_[1], 1] 
    if sleep > 4000:
        data[index_[0], index_[1], 3] -= 500
    data__.iternext()

data[:,:,2][data[:,:,2] > 9] = 9.0

avg_sleep = np.sum(data[:,:,2], axis=1)
print(f"average sleep per user: {avg_sleep}")

sum_calories = np.sum(data[:,:,3], axis=0)
print(f" sum of calories of all users per day: {sum_calories}")

high_steps = np.argmax(data[:,:,1], axis=1)
print(f"day per user with high number of steps: {high_steps}")


    

