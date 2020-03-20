import sys
import math
from enum import Enum
from typing import NewType, Optional, Tuple, List, Dict

import numpy as np

G_DEBUG = False

EntityId = NewType('EntityId', int)


class EntityType(Enum):
    MY_ROBOT = 0
    ENEMY_ROBOT = 1
    MY_RADAR = 2
    MY_TRAP = 3


class Item(Enum):
    NOTHING = -1
    RADAR = 2
    TRAP = 3
    ORE = 4


class Map:

    def __init__(self):

        def init_distance_matrix():
            # matrix of distaces beetween cells with size hw x hw, max dist = 33, so uint8
            # todo: optimize - decrease size by 4
            distance_matrix = np.zeros((height * width, height * width), dtype=np.float16)
            for i in range(height):
                for j in range(width):
                    for h in range(height):
                        for w in range(width):
                            dst = (i - h) * (i - h) + (j - w) * (j - w)
                            distance_matrix[i * width + j, h * width + w] = dst
            return np.sqrt(distance_matrix)

        width, height = [int(i) for i in input().split()]
        self.width = width
        self.height = height
        self.distance_matrix = init_distance_matrix()

        self.ore_dim, self.hole_dim = 0, 1
        map_shape = (2, height, width)
        self.map_info = np.zeros(shape=map_shape, dtype=np.int8)  # 3rd dim for ore (0) and holes(1)
        self.ore_map = self.map_info[self.ore_dim]
        self.hole_map = self.map_info[self.hole_dim]
        self.map_info[self.ore_dim] = -1

        self.radar_map = np.zeros((height, width))
        self.trap_map = np.zeros((height, width))
        self.myholes_map = np.zeros((height, width))

        self.tick = 0  # should be in Manager, but design is bad

    def _get_dist(self, x0, y0, x1, y1):
        return self.distance_matrix[y0 * self.width + x0][y1 * self.width + x1]

    def _get_dist_weighted(self, x0, y0, x1, y1):
        return self.distance_matrix[y0 * self.width + x0][y1 * self.width + x1] / self.ore_map[y1, x1]

    def get_nearest_ore(self, x, y, fltr=None, weighted=True):
        dst_f = self._get_dist_weighted if weighted else self._get_dist
        mn, x_ore, y_ore = 100, -1, -1
        for i in range(self.height):
            for j in range(self.width):
                if fltr and not fltr(i, j):
                    continue
                if self.ore_map[i, j] > 0 and self.trap_map[i, j] == 0:
                    dst = dst_f(x, y, j, i)
                    if dst < mn:
                        mn, x_ore, y_ore = dst, j, i
        return x_ore, y_ore

    def get_nearest_point(self, x, y, points):
        mn, x_mn, y_mn = 100, -1, -1
        for (i, j) in points:
            dst = self._get_dist(x, y, i, j)
            if dst < mn:
                mn, x_mn, y_mn = dst, i, j
        return x_mn, y_mn

    def _in_bounds(self, x, y):
        if x < 0 or x > 29:
            return False
        if y < 0 or y > 14:
            return False
        return True

    def get_adjecent_vein(self, x, y):
        coord = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        adjs = [(x + dx, y + dy) for (dx, dy) in coord if
                self._in_bounds(x + dx, y + dy) and not self.hole_map[y + dy, x + dx]]
        ores = [self.ore_map[yy, xx] for (xx, yy) in adjs]
        if not len(ores) or max(ores) == 0:
            return -1, -1
        else:
            return adjs[np.argmax(ores)]

    def update_state(self):
        cur_map_info = np.array([input().split() for _ in range(self.height)]).reshape((self.height, self.width, 2))
        cur_map_info = np.moveaxis(cur_map_info, -1, 0)
        ore_selector = cur_map_info[self.ore_dim] != '?'
        self.ore_map[ore_selector] = cur_map_info[self.ore_dim][ore_selector].astype(np.uint8)
        ore_selector = cur_map_info[self.ore_dim] == '?'
        self.ore_map[ore_selector] = 0
        self.map_info[self.hole_dim] = cur_map_info[self.hole_dim].astype(
            np.int8)  # (self.map_info[self.hole_dim] + cur_map_info[self.hole_dim].astype(np.int8)) % 2

    def update_support_maps(self, radars_coords: List[Tuple[int, int]], traps_coords: List[Tuple[int, int]]):
        self.radar_map[:] = 0
        self.trap_map[:] = 0
        for (x, y) in radars_coords:
            self.radar_map[y, x] = 1
        for (x, y) in traps_coords:
            self.trap_map[y, x] = 1


class Entity:
    def __init__(self, entity_id: int, entity_type: int, x: int, y: int, item: int):
        self.entity_id: EntityId = EntityId(entity_id)
        self.entity_type: EntityType = EntityType(entity_type)
        self.x: int = x
        self.y: int = y
        self.item: Item = Item(item)

    def update_state(self, x: int, y: int, item: int):
        self.x = x
        self.y = y
        self.item = Item(item)


class Robot(Entity):
    def __init__(self, entity_id: int, entity_type: int, x: int, y: int, item: int):
        if (entity_type != EntityType.MY_ROBOT.value
                and entity_type != EntityType.ENEMY_ROBOT.value):
            raise ValueError(f'entity type {entity_type} is not a robot identifier')
        super().__init__(entity_id, entity_type, x, y, item)
        self.is_dead: bool = False

    def update_state(self, x: int, y: int, item: int):
        super().update_state(x, y, item)
        if self.x == -1:
            self.is_dead = True
        # self.command = Command.wait('updated')

    def has_item(self, item: Item) -> bool:
        return self.item == item

    def on_base(self):
        return self.x == 0


class MyRobot(Robot):
    def __init__(self, entity_id: int, entity_type: int, x: int, y: int, item: int):
        super().__init__(entity_id, entity_type, x, y, item)
        self.command: str = 'WAIT initialized'

    def update_state(self, x: int, y: int, item: int):
        super().update_state(x, y, item)
        if self.is_dead:
            self.wait(f'{self.entity_id} is dead')
        # self.command = Command.wait('updated')

    def wait(self, comment: str = ''):
        self.command = f'WAIT # {comment}'

    def move(self, x: int, y: int, comment: str = ''):
        self.command = f'MOVE {x} {y} # {comment}'

    def dig(self, x: int, y: int, comment: str = ''):
        self.command = f'DIG {x} {y} # {comment}'

    def request(self, item: Item, comment: str = ''):
        self.command = f'REQUEST {item.name} # {comment}'

    def print_command(self):
        cmd = self.command.replace('# ', '') if G_DEBUG else self.command.split('#')[0]
        print(cmd)


class Manager:

    def __init__(self, game_map: Map):
        self.map: Map = game_map
        self.entities: Dict[EntityId, Entity] = dict()
        self.myrobots: Dict[EntityId, MyRobot] = dict()
        self.radar_cooldown: int = 0
        self.trap_cooldown: int = 0
        self.my_score: int = 0
        self.opponent_score: int = 0
        self.received_ids: List[EntityId] = []

    def update_state(self, my_score: int, opponent_score: int):
        self.my_score = my_score
        self.opponent_score = opponent_score
        entity_count, radar_cooldown, trap_cooldown = [int(i) for i in input().split()]
        self.radar_cooldown = radar_cooldown
        self.trap_cooldown = trap_cooldown
        self.received_ids = []
        radars_coordinates = []
        trap_coordinates = []
        for i in range(entity_count):
            entity_id, entity_type, x, y, item = [int(j) for j in input().split()]
            if entity_id not in self.entities:
                if entity_type == EntityType.MY_ROBOT.value:
                    entity = MyRobot(entity_id, entity_type, x, y, item)
                    self.myrobots[entity_id] = entity
                elif entity_type == EntityType.ENEMY_ROBOT.value:
                    entity = Robot(entity_id, entity_type, x, y, item)
                else:
                    entity = Entity(entity_id, entity_type, x, y, item)
                self.entities[entity_id] = entity
            else:
                self.entities[entity_id].update_state(x, y, item)

            self.received_ids.append(entity_id)

            entity = self.entities[entity_id]
            if entity.entity_type == EntityType.MY_RADAR:
                radars_coordinates.append((entity.x, entity.y))
            if entity.entity_type == EntityType.MY_TRAP:
                trap_coordinates.append((entity.x, entity.y))
        self.map.update_support_maps(radars_coordinates, trap_coordinates)

    def tick(self):
        # print commands
        for eid in self.received_ids:
            if eid in self.myrobots:
                self.myrobots[eid].print_command()
        self.map.tick += 1


class GreedyManager(Manager):

    def __init__(self, game_map: Map):
        super().__init__(game_map)
        self.scout: Optional[MyRobot] = None
        self.scout_status_changed = True
        self.route = [(5, 4), (5, 10), (10, 7), (15, 4), (15, 10), (10, 0),
                      (10, 14), (20, 7), (25, 4), (25, 10), (20, 0), (20, 14)]
        self.harvesters: List[MyRobot] = []
        self.robots_strategies_map: Dict[EntityId, MyRobotStrategy] = dict()

    def update_state(self, my_score: int, opponent_score: int):
        super().update_state(my_score, opponent_score)
        if not self.harvesters:
            self.harvesters = [robot for robot in self.myrobots.values()]

    def _keep_alive(self):
        if self.scout and self.scout.is_dead:
            self.scout = None
        robots_to_remove = []  # todo: do better
        for robot in self.harvesters:
            if robot.is_dead:
                robots_to_remove.append(robot)
        for robot in robots_to_remove:
            self.harvesters.remove(robot)

    def _scout(self):
        if not self.scout:
            self.scout = self.harvesters.pop()
            self.scout.request(Item.RADAR, 'no scouts')
            self.scout_status_changed = True
        else:
            if not self.scout.has_item(Item.RADAR) and self.scout_status_changed:
                self.scout.request(Item.RADAR, 'no radar')
                self.scout_status_changed = False
                x = self.route.pop(0)
                self.route.append(x)  # todo Value Error
            elif self.scout.has_item(Item.RADAR):
                x, y = self.route[0]
                self.scout.dig(x, y, 'place radar')
                self.scout_status_changed = True

    def _harvest(self):
        for harvester in self.harvesters:
            # self.robots_strategies_map[harvester.entity_id] = HarvestStrategy(self.map)
            if harvester.has_item(Item.ORE):
                harvester.move(0, harvester.y, 'deliver ore')
            else:
                try:
                    x_ore, y_ore = self.map.get_nearest_ore(harvester.x, harvester.y)
                    harvester.dig(x_ore, y_ore, 'digging nearest ore')
                except ValueError:
                    harvester.wait('Waiting for scouting')

    def tick(self):
        self._keep_alive()
        self._scout()
        self._harvest()

        super().tick()


class SmartManager(Manager):
    def __init__(self, game_map: Map):
        super().__init__(game_map)
        self.robots_strategies_map: Dict[EntityId, MyRobotStrategy] = dict()

    def tick(self):
        # self.robots_strategies_map = dict()

        # todo: dead robots
        # update strategy for current scouters
        for eid, strategy in self.robots_strategies_map.items():
            robot = self.myrobots[eid]
            if isinstance(strategy, ScoutStrategy) and not robot.has_item(Item.RADAR):
                self.robots_strategies_map[eid] = HarvestStrategy(self.map, 'nearest_nonenemy')

        robots_on_base = [robot for robot in self.myrobots.values() if robot.on_base()]
        empty_robots_on_base = [robot for robot in robots_on_base if robot.has_item(Item.NOTHING)]
        robots_on_base = empty_robots_on_base
        is_all_placed = np.all(ScoutStrategy.all_placed(self.map.radar_map))
        if self.radar_cooldown == 0 and not is_all_placed and len(robots_on_base) > 0:
            eid = robots_on_base.pop().entity_id
            self.robots_strategies_map[eid] = ScoutStrategy(self.map)

        if self.trap_cooldown == 0 and len(robots_on_base) > 0:
            eid = robots_on_base.pop().entity_id
            self.robots_strategies_map[eid] = MineStrategy(self.map)

        robots_on_base = [robot for robot in robots_on_base if robot.has_item(Item.NOTHING)]
        for robot in robots_on_base:
            eid = robot.entity_id
            self.robots_strategies_map[eid] = HarvestStrategy(self.map, 'nearest_nonenemy')

        # execute strategies
        for eid, strategy in self.robots_strategies_map.items():
            strategy.execute(self.myrobots[eid])

        super().tick()


class MyRobotStrategy:

    def __init__(self, game_map: Map):
        self.map = game_map

    def execute(self, robot: MyRobot):
        pass

    def _dig(self, robot: MyRobot, x: int, y: int, comment: str = ''):
        """
        control holes made by me
        :return:
        """
        dx, dy = abs(robot.x - x), abs(robot.y - y)
        self.map.hole_map[y, x] = 1  # mark for other robots as uninterested
        if dx + dy == 1:
            self.map.myholes_map[y, x] = 1
            self.map.hole_map[y, x] = 1
            self.map.ore_map[y, x] -= 1
        robot.dig(x, y, f'{(robot.x, robot.y)}  ' + comment)


class HarvestStrategy(MyRobotStrategy):
    def __init__(self, game_map: Map, method: str = 'nearest'):
        super().__init__(game_map)
        if method not in {'nearest', 'nearest_nonenemy'}:
            raise ValueError(f'{method} is not supported')
        self.method = method

    def execute(self, robot: MyRobot):
        x_ore, y_ore = self.map.get_adjecent_vein(robot.x, robot.y)
        if x_ore != -1:
            return super()._dig(robot, x_ore, y_ore, 'mark nearest vein')

        if robot.has_item(Item.ORE):
            robot.move(0, robot.y, 'deliver ore')
        else:
            fltr = None
            if self.method == 'nearest':
                fltr = None
                x_ore, y_ore = self.map.get_nearest_ore(robot.x, robot.y, fltr)
            elif self.method == 'nearest_nonenemy':  # w/ priority
                x_ore = -1
                if self.map.tick < 160:
                    fltr = lambda i, j: not self.map.hole_map[i, j]
                    x_ore, y_ore = self.map.get_nearest_ore(robot.x, robot.y, fltr)
                if x_ore < 0:  # w/o priority
                    fltr = lambda i, j: self.map.myholes_map[i, j] == 1 and self.map.hole_map[i, j] or not \
                    self.map.hole_map[i, j]
                    x_ore, y_ore = self.map.get_nearest_ore(robot.x, robot.y, fltr)
            if x_ore > 0:
                super()._dig(robot, x_ore, y_ore, 'digging nearest ore')
            else:  # then just move a little bit forward
                robot.move(0, 7, 'moving forward')  # todo: move forward
        pass


class ScoutStrategy(MyRobotStrategy):
    radar_points = np.array([(10, 7), (5, 4), (5, 10), (15, 4), (15, 10),
                             (20, 7), (10, 0), (10, 14), (25, 4), (25, 10),
                             (20, 0), (20, 14), (27, 2), (27, 12), (27, 7)]).reshape(-1, 2)

    def __init__(self, game_map: Map, method: str = 'order'):
        super().__init__(game_map)
        if method not in {'nearest', 'order'}:
            raise ValueError(f'{method} is not supported')
        self.method = method
        self.x_r, self.y_r = 0, 0

    @staticmethod
    def all_placed(radar_map):
        radars_coords = np.argwhere(radar_map == 1)
        radars_exist = np.zeros(ScoutStrategy.radar_points.shape[0], dtype=bool)
        for i in range(ScoutStrategy.radar_points.shape[0]):
            for (y, x) in radars_coords:
                if ScoutStrategy.radar_points[i][0] == x and ScoutStrategy.radar_points[i][1] == y:
                    radars_exist[i] = True
        return radars_exist

    def execute(self, robot: MyRobot):
        if not robot.has_item(Item.RADAR):
            robot.request(Item.RADAR, 'has no radar')
        else:
            radars_exist = self.all_placed(self.map.radar_map)
            if self.method == 'nearest':
                x_r, y_r = self.map.get_nearest_point(robot.x, robot.y, self.radar_points[~radars_exist])
                if x_r < 0:
                    return HarvestStrategy(self.map, 'nearest_nonenemy').execute(robot)
            elif self.method == 'order':
                if self.radar_points[~radars_exist].shape[0] > 0:
                    is_old_target = np.any(np.all(self.radar_points[~radars_exist] == [self.x_r, self.y_r], axis=1))
                    if is_old_target:
                        x_r, y_r = self.x_r, self.y_r
                    else:
                        x_r, y_r = self.radar_points[~radars_exist][0]
                else:
                    return HarvestStrategy(self.map, 'nearest_nonenemy').execute(robot)

            self.x_r, self.y_r = x_r, y_r
            super()._dig(robot, x_r, y_r, 'place radar')


class MineStrategy(MyRobotStrategy):
    def __init__(self, game_map: Map, method: str = 'nearest'):
        super().__init__(game_map)
        if method not in {'nearest'}:
            raise ValueError(f'{method} is not supported')
        self.method = method

    def execute(self, robot: MyRobot):
        if not robot.has_item(Item.TRAP):
            robot.request(Item.TRAP, 'has no trap')
        else:
            selector = (self.map.ore_map == 2) & (self.map.trap_map == 0)
            enemyholes = (self.map.myholes_map == 0) & (self.map.hole_map == 1)
            selector &= ~enemyholes
            no_holes = ~ (self.map.hole_map == 1)
            potential_spots = np.argwhere(selector & no_holes)[:, ::-1]
            if potential_spots.shape[0] == 0:
                potential_spots = np.argwhere(selector)[:, ::-1]
            if potential_spots.shape[0] == 0:
                return HarvestStrategy(self.map, 'nearest_nonenemy').execute(robot)
            if self.method == 'nearest':
                x_r, y_r = self.map.get_nearest_point(robot.x, robot.y, potential_spots)
            if x_r < 0:
                return HarvestStrategy(self.map, 'nearest_nonenemy').execute(robot)
            super()._dig(robot, x_r, y_r, 'place trap')


def main():
    game_map = Map()
    strategy = SmartManager(game_map)
    while True:
        my_score, opponent_score = [int(i) for i in input().split()]
        game_map.update_state()
        strategy.update_state(my_score, opponent_score)
        strategy.tick()

main()
