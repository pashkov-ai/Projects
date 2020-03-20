from problems.p849 import Solution

sol = Solution()


def test_ex1():
    seats = [1, 0, 0, 0, 1, 0, 1]
    assert sol.maxDistToClosest(seats) == 2


def test_ex2():
    seats = [1, 0, 0, 0]
    assert sol.maxDistToClosest(seats) == 3


def test_ex3():
    seats = [0, 0, 0, 1]
    assert sol.maxDistToClosest(seats) == 3


def test_ex4():
    seats = [0, 0, 1, 0, 0, 1]
    assert sol.maxDistToClosest(seats) == 2


def test_ex5():
    seats = [0, 1, 0, 0, 0, 1]
    assert sol.maxDistToClosest(seats) == 2


def test_ex6():
    seats = [1, 0, 0, 1]
    assert sol.maxDistToClosest(seats) == 1
