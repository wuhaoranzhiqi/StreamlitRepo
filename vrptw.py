import copy
import re
import math

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
import pandas as pd

from alns import ALNS, State
from alns.accept import RecordToRecordTravel
from alns.select import RouletteWheel
from alns.stop import MaxRuntime

SEED = 1234
Speed = 40

class Data():
    def __init__(self):
        self.name = ""
        self.n_customers = 0
        self.dimension = 0
        
        self.vehicle_kinds = 0
        self.vehicle_types = []
        self.vehicle_num = {}
        self.vehicle_fix_cost = {}
        self.vehicle_var_cost = {}
        self.vehicle_capacity = {}


        self.coordinates = []
        self.demands = []
        self.readyTime = []
        self.dueTime = []
        self.serviceTime = []

        self.distances = [[]]
        
        self.used_vehicle = {}


def haversine(lon1, lat1, lon2, lat2):
        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (np.sin(dlat / 2)**2 +
             np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2)
        c = 2 * np.arcsin(np.sqrt(a))

        # 6367 km is the radius of the Earth
        km = 6367 * c
        return km

def readData(data, path, customerNum):
    data.n_customers = customerNum
    data.dimension = customerNum + 1
    f = open(path, 'r')
    lines = f.readlines()
    count = 0
    
    for line in lines:
        count += 1
        if (count == 1):
            data.vehicle_kinds = 3
            data.vehicle_types = ["v1", "v2", "v3"]
            data.vehicle_num = {"v1": 5, "v2": 8, "v3": 2}
            data.vehicle_fix_cost = {"v1": 0, "v2": 0, "v3": 0}
            data.vehicle_var_cost = {"v1": 2, "v2": 4, "v3": 3}
            data.vehicle_capacity = {"v1": 18, "v2": 32, "v3": 26}
#             data.greedy_vehicle_num = {"v1": 5, "v2": 8, "v3": 3}
        elif (count >= 2 and count <= 2 + customerNum):
            line = line[:-1]
            str = re.split(r",", line)
            data.coordinates.append([float(str[1]), float(str[2])])
            data.demands.append(float(str[3]))
            data.readyTime.append(float(str[4]))
            data.dueTime.append(float(str[5]))
            data.serviceTime.append(float(str[6]))
    
    data.distances = [([0] * data.dimension) for p in range(data.dimension)]
    for i in range(data.dimension):
        for j in range(data.dimension):
#             temp = (data.coordinates[i][0] - data.coordinates[j][0]) ** 2 + (data.coordinates[i][1] - data.coordinates[j][1]) ** 2
            dist = haversine(data.coordinates[i][0],
                             data.coordinates[i][1],
                             data.coordinates[j][0],
                             data.coordinates[j][1]) 
            data.distances[i][j] = dist / Speed
    
    for v in data.vehicle_types:
        data.used_vehicle.update({v: 0})
    print(data.used_vehicle)
    
    return data

class CvrpState(State):
    """
    Solution state for CVRP. It has two data members, routes and unassigned.
    Routes is a list of list of integers, where each inner list corresponds to
    a single route denoting the sequence of customers to be visited. A route
    does not contain the start and end depot. 
    Unassigned is a list of integers, each integer representing an unassigned customer.
    """
    
    def __init__(self, routes, unassigned=None):
        self.routes = routes
        self.unassigned = unassigned if unassigned is not None else []

    def copy(self):
        # 负责新一轮迭代, 所以去掉车辆类型
        copy_routes = copy.deepcopy(self.routes)
        for route in copy_routes:
            del route[0]
        return CvrpState(copy_routes, self.unassigned.copy())

    def objective(self):
        """
        Computes the total route costs.
        """
        return sum(route_cost(route) for route in self.routes)
    
    @property
    def cost(self):
        """
        Alias for objective method. Used for plotting.
        """
        return self.objective()

    def find_route(self, customer):
        """
        Return the route that contains the passed-in customer.
        """
        for route in self.routes:
            if customer in route:
                return route

        raise ValueError(f"Solution does not contain customer {customer}.")
        
def route_cost(route):
    cost = 0
    r_customers = route[1:]
    v = route[0]
    
    if (len(route) <= 1):
        return cost
    else:
         # loadviolation cost
        total_load = sum(data.demands[cust] for cust in r_customers)
        if total_load > data.vehicle_capacity[v]:
            load_cost = (total_load - data.vehicle_capacity[v]) * 1000
            cost += load_cost

        # timeviolation cost
        departure = 0
        pre = 0
        for i in range(len(r_customers)):
            curr = r_customers[i]
            arrival = max(departure + data.distances[pre][curr], data.readyTime[curr])
            departure = arrival + data.serviceTime[curr]

            ### 判断时间窗口条件
            if arrival > data.dueTime[curr]:
                time_cost = (arrival - data.dueTime[curr]) * 1000
                cost += time_cost
            pre = curr
            ## 判断最后到depot点的时间窗
            if curr == r_customers[-1]:
                if departure + data.distances[curr][0] > data.dueTime[0]:
                    time_cost = (departure + data.distances[curr][0] - data.dueTime[0]) * 1000
                    cost += time_cost
    
    tour = [0] + r_customers + [0]
    dis_cost = sum(data.distances[tour[idx]][tour[idx + 1]] for idx in range(len(tour) - 1))
    cost += data.vehicle_fix_cost[v] + dis_cost * data.vehicle_var_cost[v]
    return cost

def greedy_repair(state, rnd_state):
    """
    Inserts the unassigned customers in the best route. If there are no
    feasible insertions, then a new route is created.
    """
    for key in data.vehicle_types:
        data.used_vehicle[key] = 0

    for route in state.routes:
        mincost = 99999999
        min_v = None
        for v in data.vehicle_types:
            demand = 0
            # 判断该车型是否有车辆剩余
            if data.used_vehicle[v] >= data.vehicle_num[v]:
                continue
            # 判断该路径是否超过该车型的容量
            demand = sum(data.demands[cust] for cust in route)
            if data.vehicle_capacity[v] >= demand:
                v_route = [v] + route
                cost = route_cost(v_route)
            else:
                cost = (demand - data.vehicle_capacity[v]) * 10000
            if cost <= mincost:
                mincost = cost
                min_v = v
        route.insert(0, min_v)
        #print(min_v)
        data.used_vehicle[min_v] += 1
#     print("<<<<<<<<<<<<<<<<<添加车辆后routes>>>>>>>>>>>>>>>>>>>>>>")
#     print(state.routes)
#     print("<<<<<<<<<<<<<<<<<共使用车辆>>>>>>>>>>>>>>>>>>>>>>")
#     print(data.used_vehicle)
    
    rnd_state.shuffle(state.unassigned)
#     print("开始修复——————————————破化后的routes")
#     print(state.routes)
#     print("开始修复——————————————打乱后未分配的顾客点")
#     print(state.unassigned)
    
    while len(state.unassigned) != 0:
        customer = state.unassigned.pop()
        route, idx = best_insert(customer, state)

        #print(route)
#         if route[0] != veh:
#             data.used_vehicle[route[0]] -= 1
#             data.used_vehicle[veh] += 1
        #print("最优车辆+ruote+cus")
        # print(v)
        # print(route)
        route.insert(idx, customer)
#         route[0] = veh
        #print(v, route, customer)
#     print("<<<<<<<<<<<<<<<<<修复后的routes>>>>>>>>>>>>>>>>>>>>>>")
#     print(state.routes)
#     print("<<<<<<<<<<<<<<<<<本轮修复完成>>>>>>>>>>>>>>>>>>>>>>")

    return state

def best_insert(customer, state):
    """
    Finds the best feasible route and insertion idx for the customer.
    Return (None, None) if no feasible route insertions are found.
    """
    best_cost, best_route, best_idx = None, None, None

    for route in state.routes:
        # print(v)
        "route[车型，depot，。。。]"
        for idx in range(1,len(route) + 1):
            # print(idx)
            cost = insert_cost(customer, route, idx)
            if best_cost is None or cost < best_cost:
                best_cost, best_route, best_idx = cost, route, idx

    return best_route, best_idx

def insert_cost(customer, route, idx):
    """
    Computes the insertion cost for inserting customer in route at idx.
    """
    
    # distance cost
    v = route[0]
    r_customers = route[1:]
    pred = 0 if idx == 1 else route[idx - 1]
    succ = 0 if idx == len(route) else route[idx]
    # print(v,customer,idx,pred,succ)
    # print(route)
    # print(data.vehicle_cost[v],data.distances[pred][customer],data.distances[customer][succ])
    cost = data.vehicle_var_cost[v] * (data.distances[pred][customer] + data.distances[customer][succ])
    cost -= data.vehicle_var_cost[v] * data.distances[pred][succ]

    if (data.used_vehicle[v] + 1) > data.vehicle_num[v]:
        cost += (data.used_vehicle[v] + 1 - data.vehicle_num[v]) * 1000
    
    # loadviolation cost
    total_load = sum(data.demands[cust] for cust in r_customers) + data.demands[customer]
    if total_load > data.vehicle_capacity[v]:
        load_cost = (total_load - data.vehicle_capacity[v]) * 1000
        cost += load_cost
    
    # timeviolation cost
    route.insert(idx, customer)
    departure = 0
    pre = 0
    for i in range(len(r_customers)):
        curr = r_customers[i]
        arrival = max(departure + data.distances[pre][curr], data.readyTime[curr])
        departure = arrival + data.serviceTime[curr]

        ### 判断时间窗口条件
        if arrival > data.dueTime[curr]:
            time_cost = (arrival - data.dueTime[curr]) * 1000
            cost += time_cost
        pre = curr
        ## 判断最后到depot点的时间窗
        if curr == r_customers[-1]:
            if departure + data.distances[curr][0] > data.dueTime[0]:
                time_cost = (departure + data.distances[curr][0] - data.dueTime[0]) * 1000
                cost += time_cost
    _ = route.pop(idx)
    return cost

def neighbors(customer):
    """
    Return the nearest neighbors of the customer, excluding the depot.
    """
    locations = np.argsort(data.distances[customer])
    return locations[locations != 0]


def nearest_neighbor():
    """
    Build a solution by iteratively constructing routes, where the nearest
    customer is added until the route has met the vehicle capacity limit.
    """
    routes = []
#     vehicles = []
    unvisited = list(range(1, data.dimension))
    V_index = 0
    V = data.vehicle_types[0]
    VC = data.vehicle_capacity[V]

    while unvisited:
        if data.used_vehicle[V] >= data.vehicle_num[V]:
            V_index += 1
            V = data.vehicle_types[V_index]
            VC = data.vehicle_capacity[V]

        data.used_vehicle[V] += 1
        
        route = [0]  # Start at the depot
        route_demands = 0
        departure = 0    # 离开时间
        
        while unvisited:
            # Add the nearest unvisited customer to the route till max capacity
            current = route[-1]
            nearest = [nb for nb in neighbors(current) if nb in unvisited][0]
            
            arrival = max(departure + data.distances[current][nearest], data.readyTime[nearest])
            departure = arrival + data.serviceTime[nearest]
            
            ### 判断条件
            if (route_demands + data.demands[nearest] > VC):
                break
            if (arrival > data.dueTime[nearest]) or (departure + data.distances[nearest][0] > data.dueTime[0]):
                break

            route.append(nearest)
            unvisited.remove(nearest)
            route_demands += data.demands[nearest]

        customers = route[1:]  # Remove the depot
        customers.insert(0, V)
        routes.append(customers)

    return CvrpState(routes)


MAX_STRING_REMOVALS = 2
MAX_STRING_SIZE = 12

def string_removal(state, rnd_state):
    """
    Remove partial routes around a randomly chosen customer.
    """
    destroyed = state.copy()

    avg_route_size = int(np.mean([len(route) for route in state.routes]))
    max_string_size = max(MAX_STRING_SIZE, avg_route_size)
    max_string_removals = min(len(state.routes), MAX_STRING_REMOVALS)

    destroyed_routes = []
    center = rnd_state.randint(1, data.dimension)

    for customer in neighbors(center):
        if len(destroyed_routes) >= max_string_removals:
            break

        if customer in destroyed.unassigned:
            continue

        route = destroyed.find_route(customer)
        if route in destroyed_routes:
            continue

        customers = remove_string(route, customer, max_string_size, rnd_state)
        destroyed.unassigned.extend(customers)
        destroyed_routes.append(route)

    return destroyed


def remove_string(route, cust, max_string_size, rnd_state):
    """
    Remove a string that constains the passed-in customer.
    """
    # Find consecutive indices to remove that contain the customer
    size = rnd_state.randint(1, min(len(route), max_string_size) + 1)
    start = route.index(cust) - rnd_state.randint(size)
    idcs = [idx % len(route) for idx in range(start, start + size)]

    # Remove indices in descending order
    removed_customers = []
    for idx in sorted(idcs, reverse=True):
        removed_customers.append(route.pop(idx))

    return removed_customers

data = Data()
path = "data/运力平台测试数据/demand.csv"

def getPos(result):
    path_dict = {}
    index = 0
    for route in result.best_state.routes:
        if len(route) > 0:
            res = []
            res.append(data.coordinates[0])
            for cus in route[1:]:
                res.append(data.coordinates[cus])
            res.append(data.coordinates[0])
            path_dict[index] = res
            index += 1
    return pd.Series(path_dict)

def run():
    readData(data, path, 50)
    alns = ALNS(rnd.RandomState(SEED))
    alns.add_destroy_operator(string_removal)
    alns.add_repair_operator(greedy_repair)

    init = nearest_neighbor()
    select = RouletteWheel([25, 5, 1, 0], 0.8, 1, 1)
    accept = RecordToRecordTravel.autofit(init.objective(), 0.02, 0, 8000)
    stop = MaxRuntime(10)

    result = alns.iterate(init, select, accept, stop)
    paths = getPos(result)

    return paths
