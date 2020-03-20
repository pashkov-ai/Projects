"""
554. Brick Wall
Medium

There is a brick wall in front of you. The wall is rectangular and has several rows of bricks. The bricks have the same height but different width. You want to draw a vertical line from the top to the bottom and cross the least bricks.

The brick wall is represented by a list of rows. Each row is a list of integers representing the width of each brick in this row from left to right.

If your line go through the edge of a brick, then the brick is not considered as crossed. You need to find out how to draw the line to cross the least bricks and return the number of crossed bricks.

You cannot draw a line just along one of the two vertical edges of the wall, in which case the line will obviously cross no bricks.



Example:

Input: [[1,2,2,1],
        [3,1,2],
        [1,3,2],
        [2,4],
        [3,1,2],
        [1,3,1,1]]

Output: 2

Explanation:



Note:

    The width sum of bricks in different rows are the same and won't exceed INT_MAX (W).
    The number of bricks in each row is in range [1,10,000] (R). The height of wall is in range [1,10,000] (H). Total number of bricks of the wall won't exceed 20,000 (N).
"""
from typing import List


# time: O(HR + W) || O(N + W), space: O(W)
class Solution:
    def leastBricks(self, wall: List[List[int]]) -> int:
        joint_offsets = dict()  # O(W) - space
        for row in wall:  # O(H) - time
            cum_off = 0
            for i in range(len(row) - 1):  # O(R) - time
                cum_off += row[i]
                joint_offsets[cum_off] = joint_offsets.get(cum_off, 0) + 1
        if len(joint_offsets.values()) == 0:
            return len(wall)
        return len(wall) - max(joint_offsets.values())  # O(W) - time

        # return len(wall) - max(joint_offsets.values(), default=0)
