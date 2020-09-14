import numpy as np
import random, operator
import pandas as pd
import matplotlib.pyplot as plt

# total time: 20s
# mean velocity: 40 km/h = 11.1 m/s
# total distance: 220 m
# v_begin = 8:14.2:0.31 m/s
# v_end = 14.2:8:-0.31 m/s
# acc = (22.2 - 8 - 8) / 20 m/s^2

# Vehicle parameters
C_d = 0.55  # Wind resistance
A_d = 5.69  # Wind area / m^2
g_f = 4.11  # Final reduction drive ratio
r = 0.775 / 2  # Wheel radius / m
g = 9.81  # Gravity acceleration
mu = 0.01  # rolling resistance
air_density = 1.2
rot_coef = 1.05
m = 3200  # Vehicle mass
motor_eff = 0.91
driveline_eff = 0.92
F_u = m * g * mu
t_window = 1.0

# Create necessary classes and functions

# Create class to handle each velocity point
class DrivingPoint:
    # t: (t)th second
    # acc: acceleration between (t-1)th and (t)th second
    # vel: velocity at (t)th second
    def __init__(self, t, acc, vel=0.0):
        self.t = t
        if acc >= 0:
            self.acc = min(3.5, acc)
        else:
            self.acc = max(-3.5, acc)
        self.vel = vel

    # IMPORTANT
    # Call this first to update current velocity
    # IMPORTANT
    def updateCurrentVelocity(self, DrivingPoint):
        # Use previous driving point's velocity and current acc to update current velocity
        self.vel = self.acc + DrivingPoint.vel
        return self.vel

    def distanceToPreviousPoint(self, DrivingPoint):
        self.updateCurrentVelocity(DrivingPoint)
        v_mean = (DrivingPoint.vel + self.vel) / 2.0
        distance = v_mean * 1.0
        return distance

    def energyToPreviousPoint(self, DrivingPoint):
        self.updateCurrentVelocity(DrivingPoint)
        C_d = 0.55
        A_d = 5.69
        g_f = 4.11
        r = 0.775 / 2
        g = 9.81
        mu = 0.01
        air_density = 1.2
        rot_coef = 1.05
        m = 3200
        motor_eff = 0.91
        driveline_eff = 0.92
        F_u = m * g * mu
        t_window = 1.0
        v_mean = (self.vel + DrivingPoint.vel) / (2 * t_window)
        F_air = 0.5 * air_density * C_d * A_d * (v_mean ** 2)
        T_Req = (F_air + F_u + m * self.acc * rot_coef) * r / g_f
        nf = (v_mean * g_f * 30) / (3.14159 * r) / 3.6
        if self.acc >= 0:
            P = T_Req * nf / 9550 * motor_eff * driveline_eff
        else:
            P = -T_Req * nf / 9550 * motor_eff * driveline_eff * (0.0411 / abs(acc)) ** (-1)
        energy = P
        return energy

    def __repr__(self):
        return f't = {self.t}, acc = {self.acc}, velocity = {self.vel}'


# Create a fitness class
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.energy = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            # t = 0, vel = 8;
            # t = 20, vel = 14.2;
            # acc = 0.31
            distance_begin = self.route[0].distanceToPreviousPoint(DrivingPoint(t=0, acc=0, vel=8))
            for i in range(0, len(self.route) - 1):
                # Every member in route/self.route is an instance of DrivingPoint
                fromPoint = self.route[i]
                toPoint = self.route[i + 1]
                pathDistance += toPoint.distanceToPreviousPoint(fromPoint)
            pathDistance = pathDistance + distance_begin
            self.distance = pathDistance
        return self.distance

    # A complete route's consumed energy
    def routeEnergy(self):
        if self.energy == 0:
            pathEnergy = 0
            pathEnergy_begin = self.route[0].energyToPreviousPoint(DrivingPoint(t=0, acc=0, vel=8))
            # Every member in route/self.route is an instance of Velocity
            for i in range(0, len(self.route) - 1):
                fromPoint = self.route[i]
                toPoint = self.route[i + 1]
                pathEnergy += toPoint.energyToPreviousPoint(fromPoint)
            pathEnergy = pathEnergy + pathEnergy_begin
            self.energy = pathEnergy
        return self.energy

    # Everytime fitness is calculated, velocity is updated
    def routeFitness(self):
        # abnormal_acc_vel = [1 for i in range(0, len(self.route) - 1) if (abs(self.route[i].y - self.route[i+1].y) > 3.5)]
        # if len(abnormal_acc_vel) > 0:
        #     self.fitness = -1
        if self.fitness == 0:
            self.fitness = 100 - float(self.routeEnergy()) - max(0, abs(self.routeDistance() - 220) - 5) ** 2 \
                           - max(0, abs(self.route[19].vel - 14.2) - 1) ** 2
        return self.fitness


# Construct the genetic algorithm
# Rank individuals in descending order according to their fitness
# Each ele in fitnessResults is a tuple shaped like (index, fitness)
def rankRoutes(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)


# Create a selection function that will be used to make the list of parents routes
# popRanked is a list of tuples which are comprised of an index and its fitness
# selectionResults is a list of chosen IDs
def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    # selectionResults contain a list of route IDs
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * np.random.uniform()
        for j in range(eliteSize, len(popRanked)):
            if pick <= (100.0 - df.iat[j, 3]):
                selectionResults.append(popRanked[j][0])
                df.iat[j, 3] = 100.0
                break
    return selectionResults


# Create mating pool
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


# Create crossover function
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(np.random.uniform() * len(parent1))
    geneB = int(np.random.uniform() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    try:
        for i in range(startGene, endGene):
            childP2.append(parent2[i])
    except IndexError:
        print("parent1: ", parent1,
              "\nparent2: ", parent2,
              "\ngeneA: ", geneA,
              "\ngengB: ", geneB,
              "\nstartGene: ", startGene,
              "\nendGene: ", endGene)


    child1 = [parent2[i] for i in range(0, startGene)] + childP1 + [parent2[i] for i in range(endGene, len(parent2))]
    child2 = [parent1[i] for i in range(0, startGene)] + childP2 + [parent1[i] for i in range(endGene, len(parent1))]
    return (child1, child2)


# Create function to run crossover over full mating pool
def breedPopulation(matingpool, eliteSize, breedRate):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    if np.random.uniform() <= breedRate:
        for i in range(0, length):
            (child1, child2) = breed(pool[i], pool[len(matingpool) - i - 1])
            children.append(child1)
            children.append(child2)
    return children


# Create a mutate function
def mutate(individual, mutationRate):
    for i in range(len(individual)):
        if (np.random.uniform() < mutationRate):
            individual[i].acc = max(individual[i].acc + np.random.uniform(), -3.5)
            individual[i].acc = min(individual[i].acc, 3.5)
    return individual


# Create function to run mutation over entire population
def mutatePopulation(population, mutationRate):
    mutatedPop = []
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop



# Run the GA

# Define velocity at t=0 and t=20
# v_mean = 11.1  # 40 km/h
# v_mean = 13.9 # 50 km/h
v_mean = 11.1
v_begin = 8
v_end = v_mean * 2 - v_begin # 14.2 m/s
t_total = 20  # s
acc = (v_end - v_begin) / t_total # 0.326 m/s^2
initial_population = 250
# v_delta = (v_end - v_begin) / (initial_population - 1)

# Initialize a velocity series
Init_vel_series = [[] for i in range(initial_population)]
begin_point = DrivingPoint(t=0, acc=0, vel=8)
end_point = DrivingPoint(t=20, acc=0.31, vel=14.2)
last_point = begin_point
for i in range(initial_population):
    # randomly set acc all 20 seconds
    for j in range(1, t_total+1):  # 1 - 19 s
        current_point = DrivingPoint(t=j, acc=0.31 + np.random.uniform(-1, 1))
        current_point.updateCurrentVelocity(last_point)
        Init_vel_series[i].append(current_point)
        last_point = current_point
    last_point = begin_point

nextGeneration = Init_vel_series
eliteSize=50
mutationRate=0.05
breedRate = 0.8
generations = 100
results = []
progress = []
for i in range(100):
    popRanked = rankRoutes(nextGeneration)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(nextGeneration, selectionResults)
    children = breedPopulation(matingpool, eliteSize, breedRate)
    nextGeneration = mutatePopulation(children, mutationRate)
    progress.append(rankRoutes(nextGeneration)[0][1])
    print("This generation's best fitness: " + str(rankRoutes(nextGeneration)[0][1]))

    # Plot every route in each generation
    # results_temp = [[ele.vel for ele in ele_res] for ele_res in nextGeneration]
    # for route in results_temp:
    #     plt.plot(route)
    # plt.show()

    # Save each generation's best route
    bestRouteIndex = rankRoutes(nextGeneration)[0][0]
    bestRoute = nextGeneration[bestRouteIndex]
    results.append(bestRoute)

plt.plot(progress)
plt.ylabel('Fitness')
plt.xlabel('Generation')
plt.show()
bestRouteIndex = rankRoutes(nextGeneration)[0][0]
bestRoute = nextGeneration[bestRouteIndex]
print(bestRoute)

bestRoutes = [[ele.vel for ele in ele_res] for ele_res in results]
for route in bestRoutes:
    plt.plot(route)
plt.show()



# Calculate each route's consumed energy
v_begin = 8
v_mean = 11.1
v_end = v_mean * 2 - v_begin
initRoutes = [[ele.vel for ele in ele_init] for ele_init in Init_vel_series]
bestRoutes = [[ele.vel for ele in ele_res] for ele_res in results]
for i in range(len(bestRoutes)):
    bestRoutes[i].insert(0, v_begin)

for i in range(len(initRoutes)):
    initRoutes[i].insert(0, v_begin)

C_d = 0.55  # Wind resistance
A_d = 5.69  # Wind area
g_f = 4.11  # Final reduction drive ratio
r = 0.775 / 2
g = 9.81
mu = 0.01  # rolling resistance
air_density = 1.2
rot_coef = 1.05
m = 3200
motor_eff = 0.91
driveline_eff = 0.92
F_u = m * g * mu
# t_window = 1.0

first_energy = []
route_energy = 0
for i in range(len(initRoutes)):
    for j in range(len(initRoutes[i]) - 1):
        v_mean = (initRoutes[i][j] + initRoutes[i][j + 1]) / 2.0
        acc = initRoutes[i][j + 1] - initRoutes[i][j]
        F_air = 0.5 * air_density * C_d * A_d * (v_mean ** 2)
        T_Req = (F_air + F_u + m * acc * rot_coef) * r / g_f
        nf = (v_mean * g_f * 30) / (3.14159 * r) / 3.6
        if acc >= 0:
            P = T_Req * nf / 9550 * motor_eff * driveline_eff
        else:
            P = -T_Req * nf / 9550 * motor_eff * driveline_eff * (0.0411 / abs(acc)) ** (-1)
        route_energy += P
    first_energy.append(route_energy)
    route_energy = 0

final_energy = []
route_energy = 0
for i in range(len(bestRoutes)):
    for j in range(len(bestRoutes[i]) - 1):
        v_mean = (bestRoutes[i][j] + bestRoutes[i][j + 1]) / 2.0
        acc = bestRoutes[i][j + 1] - bestRoutes[i][j]
        F_air = 0.5 * air_density * C_d * A_d * (v_mean ** 2)
        T_Req = (F_air + F_u + m * acc * rot_coef) * r / g_f
        nf = (v_mean * g_f * 30) / (3.14159 * r) / 3.6
        if acc >= 0:
            P = T_Req * nf / 9550 * motor_eff * driveline_eff
        else:
            P = -T_Req * nf / 9550 * motor_eff * driveline_eff * (0.0411 / abs(acc)) ** (-1)
        route_energy += P
    final_energy.append(route_energy)
    route_energy = 0
print(min(final_energy))

# route = bestRoutes[bestRoutes.index(min(bestRoutes))]
# route_energy = 0
# for i in range(len(route) - 1):
#     v_mean = (route[i] + route[i+1]) / 2.0
#     acc = route[i+1] - route[i]
#     F_air = 0.5 * air_density * C_d * A_d * (v_mean ** 2)
#     T_Req = (F_air + F_u + m * acc * rot_coef) * r / g_f
#     nf = (v_mean * g_f * 30) / (3.14159 * r) / 3.6
#     if acc >= 0:
#         P = T_Req * nf / 9550 * motor_eff * driveline_eff
#     else:
#         P = -T_Req * nf / 9550 * motor_eff * driveline_eff * (0.0411 / abs(acc)) ** (-1)
#     route_energy += P
#
# plt.plot(np.array(route)*3.6)
# plt.xlabel("Time (s)")
# plt.ylabel("Velocity (m/s)")
# plt.text(x=5,y=50, s="Mean velocity: 40 km/h")
# plt.show()
