import numpy as np

np.random.seed(42)

np.set_printoptions(suppress=True, precision=2)

users, days, features = 6, 10, 3

def creaton_of_data():
    steps = np.random.normal(7500, 2500, size=(6, 10, 1))
    sleep_hours = np.random.normal(6.8, 1.0, size=(6, 10, 1))
    calories = np.random.normal(2100, 400, size=(6, 10, 1))
    data_ = np.concatenate([steps, sleep_hours, calories], axis=2)
    return data_

data = creaton_of_data()
print(data)

def appearance_of_data(data):
    print(f"the size of a data: {data.size}")
    print(f"the shape of a data: {data.shape}")
    print(f"the data type of a data: {data.dtype}")

data_info = appearance_of_data(data)

def average_steps_per_user(data):
    avg_steps = np.average(data[:,:,0], axis=1)
    print(f"average steps per user: {avg_steps}")

average_steps = average_steps_per_user(data)

def min_sleep_per_day(data):
    min_sleep = np.min(data[:,:,1], axis=0)
    print(f"minimum sleep per day: {min_sleep}")
    
min_sleep_ = min_sleep_per_day(data)

def filter_1(data):
    for user in range(data.shape[0]):
       user_days = (data[:,:,0][user] > 9000) & (data[:,:,1][user] < 6)
       print(f"\n📦 User {user} matching days:")
       print(data[user][user_days])

    
filtered_1 =filter_1(data)
    
def high_calories(data):
    high_calories_ = np.where(data[:,:,2] > 2500)
    print(f"user and day whhen calories > 2500: {high_calories_}")
    
high_calories_ = high_calories(data)

def more_steps(data):
    high_steps = np.any((data[:,:,0:1] > 9000), axis=2)
    print(f"which user's days had at least one day above 9000 steps: {data[high_steps]}")
    
high_steps = more_steps(data)

def high_sleep_hours(data):
    more_sleep = np.all((data[:,:,1] > 5), axis=1)
    print(f"user'days  who always slept more than 5 hrs: {data[more_sleep]}")
  
more_sleep = high_sleep_hours(data)

def random_weight(data):
    random = np.random.normal(size=(3,2))
    data_ = data[0,:,:]
    dot_ = np.dot(data_,random)
    print(f"the multiply using dot: {dot_}")
    data__ = data_ @ random
    print(f"multiply using @: {data__}")
    matmul_ = np.matmul(data_,random)
    print(f"multply using matmul {matmul_}")
    
multiply = random_weight(data)
    
def reshape(data):
    reshape_ = data.reshape(60,3)
    print(f"data after reshape: {reshape_}")

reshape_ = reshape(data)

def part_1__(data):
    part__1 = data[0:3,:,:]
    print(f"users 0–2: {part__1}")
    return part__1

def part_2__(data):
    part__2 = data[3:, :, :]
    print(f"users users 3–5: {part__2}")
    return part__2
    
part_1_ = part_1__(data)
part_2_ = part_2__(data)


def concatenate_(part_1,part_2):
    connect = np.concatenate([part_1,part_2], axis=0)
    print(f"part 1 and part 2 after concatenetion: {connect}")

connect_ = concatenate_(part_1_,part_2_)
    
def clip_(data):
    data[:,:,2] = np.clip(data[:,:,2], 1500, 2800)
    print(f"data after limit calories btw 1500 and 2800:{data}")
    
limit = clip_(data)

def update_step(data):
    data_ = np.nditer(data, flags=['multi_index'], op_flags=['readwrite'])
    while not data_.finished:
        index = data_.multi_index
        if (index[2] == 1) & (data[index[0], index[1], 1] < 5):
            data[index[0], index[1], 0] += 500
        data_.iternext()
    
update = update_step(data)

def all_values(data):
    data__ = np.nditer(data, flags=['multi_index'], op_flags=['readwrite'])
    while not data__.finished:
        index_ = data__.multi_index
        value_ = data__[0]
        print(f"index: {index_}, value: {value_}")
        data__.iternext()
    
values = all_values(data)

def update_step(data):
    data_ = np.nditer(data, flags=['multi_index'], op_flags=['readwrite'])
    while not data_.finished:
        index = data_.multi_index
        if (index[2] == 0) & (data[index[0], index[1], 1] < 5):
            data[index[0], index[1], 0] += 500
        data_.iternext()
    
update_ = update_step(data)
print(f" data after updating steps: {update_}")


def health_score(data):
    health_score = (data[:,:,0]/10000) + data[:,:,1] + (data[:,:,2]/2500)
    print(f"the health score per day of users : {health_score}")
    return health_score

health_score_ = health_score(data)

def best_health_score(health_score,data):
    best_ = np.argmax(health_score, axis=1)
    for i, day in enumerate(best_):
        print(f"User {i} best day ({day}) data: {data[i, day]}")

    
best_health_score_ = best_health_score(health_score_,data)
    
    

    
    
    

