import argparse
import copy
import json
import math
import os
import random
import string
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pdb

# Entities are either agents or objects with location (x-value, y-value)
class Entity:
    def __init__(self, x, y, name):
        self.x = x
        self.y = y
        self.name = str(name)

class Agent(Entity):
    def __init__(self, x, y, name, r):
        super().__init__(x, y, name)
        self.r = r

    def observe(self, objects):
        _obs = []
        for i in range(len(objects)):
            if distance(self, objects[i]) < self.r:
                _obs.append(objects[i])
        return _obs

    def obs_to_dict(self, objects):
        _obs = self.observe(objects)
        return [{'id' : obj.name, 'x' : obj.x - self.x, 'y' : obj.y - self.y, 'color' : obj.color, 'size' : obj.size} for obj in _obs]

class Object(Entity):
    def __init__(self, x, y, name, color, size):
        super().__init__(x, y, name)
        self.color = color
        self.size = size

    @classmethod
    def gen_new_object(cls, name, grid_size, base_color, color_range, base_size, size_range):
        x = np.random.uniform(0, grid_size)
        y = np.random.uniform(0, grid_size)
        min_color = base_color - color_range / 2
        max_color = base_color + color_range / 2
        min_size = base_size - size_range / 2
        max_size = base_size + size_range / 2
        color = int(round(np.random.uniform(min_color, max_color)))
        size = int(round(np.random.uniform(min_size, max_size)))
        return cls(x, y, name, color, size)

def distance(e_1, e_2):
    '''
    e is either Entity or tuple (x,y)
    '''
    x = []
    y = []
    for e in [e_1, e_2]:
        if isinstance(e, Entity):
            x.append(e.x)
            y.append(e.y)
        elif isinstance(e, tuple):
            x.append(e[0])
            y.append(e[1])

    return np.sqrt((x[0] - x[1])**2 + (y[0] - y[1])**2)

def _probabilistic_round(x):
    return int(x) + int(np.random.uniform() < x - int(x))

def total_num_objects(args):
    '''
    make sure density of objects per area is fixed
    '''
    grid_area = (args.agt_r * 6) ** 2
    _density = args.agt_view / math.pi / (args.agt_r ** 2)
    return _probabilistic_round(grid_area * _density)

def generate_uuid(prefix):
    return prefix + '_' + ''.join([np.random.choice(list(string.digits + string.ascii_letters)) for _ in range(16)])

def gen_world(args):
    agents = []
    # agent 0 is located in the center of the grid
    agents.append(Agent(args.grid_size / 2, args.grid_size / 2, 'agent_0', args.agt_r))

    scenario_list = []

    for num_shared in range(args.min_shared, args.max_shared + 1):
        print(num_shared)
        scenarios = []
        while len(scenarios) < args.num_world_each:
            assert len(agents) == 1

            objects = []
            total_num_obj = total_num_objects(args)
            while len(objects) < total_num_obj:
                obj = Object.gen_new_object(str(len(objects)), args.grid_size, args.base_color, args.color_range, args.base_size, args.size_range)
                redo = False
                # if new object is to close the previous objects, redo
                for prev_obj in objects:
                    if distance(prev_obj, obj) < args.min_dist_obj:
                        redo = True
                        break
                if redo is False:
                    # append object to the list
                    objects.append(obj)

            # restart if number of obseravable objects for agent_0 does not match
            if len(agents[0].observe(objects)) != args.agt_view:
                continue

            while True:
                x = np.random.uniform(0, args.grid_size)
                y = np.random.uniform(0, args.grid_size)
                # by default, arg.max_dist_agt = 0.5 (agents' views should overlap)
                if distance(agents[0], (x,y)) < args.max_dist_agt:
                    _agent = Agent(x, y, 'agent_1', args.agt_r)
                    break

            if len(_agent.observe(objects)) == args.agt_view and len(set(agents[0].observe(objects)).intersection(set(_agent.observe(objects)))) == num_shared:
                agents.append(_agent)

            # successfully created a scenario!
            if len(agents) == args.num_agents:
                uuid = generate_uuid('S')
                scenario = {'uuid' : uuid}
                kbs = [agent.obs_to_dict(objects) for agent in agents]
                scenario['kbs'] = kbs
                scenario['shared'] = num_shared
                scenarios.append(scenario)
                agents.pop()

        scenario_list += scenarios

    return scenario_list


def main(): 
    parser = argparse.ArgumentParser()
    # essential world parameters
    parser.add_argument('--color_range', type=int, default=150,
        help='range of color')
    parser.add_argument('--size_range', type=int, default=6,
        help='range of size')
    parser.add_argument('--base_size', type=int, default=10,
        help='range of size')
    parser.add_argument('--min_dist_obj', type=float, default=0.04,
        help='minimum distance between objects')
    parser.add_argument('--agt_view', type=int, default=7,
        help='number of objects in each agents view')
    parser.add_argument('--min_shared', type=int, default=3,
        help='minimum number of objects shared in every agents view')
    parser.add_argument('--max_shared', type=int, default=7,
        help='maximum number of objects shared in every agents view')
    parser.add_argument('--max_dist_agt', type=float, default=0.5,
        help='maximum distance between agents')
    parser.add_argument('--num_world_each', type=int, default=10,
        help='number of total worlds to be generated for each case')
    parser.add_argument('--seed', type=int, default=12,
        help='seed')

    # save parameters
    parser.add_argument('--save_path', type=str, default='data',
        help='save generated context data')
    parser.add_argument('--file_name', type=str, default='scenario.json',
        help='save file name')

    args = parser.parse_args()

    np.random.seed(args.seed)

    # current support
    args.num_agents = 2
    args.agt_r = 0.25
    args.grid_size = args.agt_r * 6 
    args.base_color = 128

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    # create scenario list
    scenario_list = gen_world(args)

    # we need to convert scenarios to SVG format for collecting data on web
    margin = 15
    svg_radius = 200
    svg_grid_size = svg_radius * 6
    web_scenario_list = []
    for scenario in scenario_list:
        web_scenario = {'uuid' : scenario['uuid']}
        web_kbs = []
        for kb in scenario['kbs']:
            web_kb = []
            for obj in kb:
                web_obj = {}
                web_obj['id'] = obj['id']
                web_obj['x'] = round(margin + svg_radius + obj['x'] * (svg_radius / args.agt_r))
                web_obj['y'] = round(margin + svg_radius + obj['y'] * (svg_radius / args.agt_r))
                web_obj['color'] = "rgb({0},{0},{0})".format(obj['color'])
                web_obj['size'] = obj['size'] # 7 ~ 13
                web_kb.append(web_obj)
            web_kbs.append(web_kb)
        web_scenario['kbs'] = web_kbs
        web_scenario['shared'] = scenario['shared']
        web_scenario_list.append(web_scenario)

    with open(os.path.join(args.save_path, args.file_name), 'w') as out_file:
        json.dump(web_scenario_list, out_file)


if __name__ == '__main__':
    main()