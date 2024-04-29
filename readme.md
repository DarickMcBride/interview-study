# DSA Study Guide for Coding Interviews

## Introduction

This guide is designed to help you prepare for coding interviews by covering essential data structures and algorithms.

## Table of Contents

- Complexity Analysis
- Data Structures
  - Arrays
  - Linked Lists
  - Stacks and Queues
  - Trees
  - Graphs
  - Hash Tables
- Algorithms
  - Sorting
  - Searching
  - Dynamic Programming
  - Backtracking
- Problem Solving Patterns

### Big O Notation

- **O(1)**: Constant time
- **O(log n)**: Logarithmic time
- **O(n)**: Linear time
- **O(n log n)**: Linearithmic time
- **O(n^2)**: Quadratic time

## Data Structures

### Arrays

- Contiguous block of memory
- Constant-time access with index
- Insertion and deletion operations

### Linked Lists

- Sequence of nodes connected by pointers
- Constant-time insertions and deletions
- No random access

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def insert_at_beginning(self, data):
        # Insert a new node at the beginning of the list
        new_node = Node(data)  # Create a new node with the given data
        new_node.next = self.head  # Link the new node to the current head
        self.head = new_node  # Update the head to the new node

    def delete_node(self, key):
        # Delete the first node with the given key
        temp = self.head  # Start with the head of the list
        # If the head node itself holds the key to be deleted
        if (temp is not None):
            if (temp.data == key):
                self.head = temp.next  # Change head to the next node
                temp = None  # Free the old head node
                return

        # Search for the key to be deleted, keep track of the previous node
        while(temp is not None):
            if temp.data == key:
                break  # Node with the key found
            prev = temp  # Save the current node as previous
            temp = temp.next  # Move to the next node

        # If the key was not present in the list
        if(temp == None):
            return  # Key not found, nothing to delete

        # Unlink the node from the linked list
        prev.next = temp.next  # Update the previous node's next to skip over the deleted node
        temp = None  # Free the node to be deleted
```

### Stacks and Queues

- **Stacks**: LIFO data structure
- **Queues**: FIFO data structure

```python
# Stack implementation using list
stack = []

# Pushing element to stack
stack.append('a')
stack.append('b')

# Popping element from stack
stack.pop()

# Queue implementation using collections.deque
from collections import deque
queue = deque()

# Enqueue element to queue
queue.append('a')
queue.append('b')

# Dequeue element from queue
queue.popleft()
```

### Trees

- Hierarchical structure
- Binary trees, BST
- Searching: DFS BFS

```python
class TreeNode:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

# Inserting into a binary search tree
# Insert a new node and return the root of the BST.
class TreeNode:
    # Constructor to create a new tree node
    def __init__(self, key):
        self.left = None  # Initialize left child as None
        self.right = None  # Initialize right child as None
        self.val = key  # Set the value of the node

# Function to insert a new node with the given value into the BST
def insert(root, val):
    # If the current node is None, create a new node with the value and return it
    if not root:
        return TreeNode(val)
    
    # If the value to insert is greater than the current node's value,
    # insert the value in the right subtree
    if val > root.val:
        root.right = insert(root.right, val)
    # If the value to insert is less than the current node's value,
    # insert the value in the left subtree
    elif val < root.val:
        root.left = insert(root.left, val)
    # Return the unchanged node pointer
    return root

# Function to find the node with the minimum value in the BST
def minValueNode(root):
    curr = root  # Start with the current node
    # Loop to find the leftmost leaf
    while curr and curr.left:
        curr = curr.left
    # The leftmost leaf will have the minimum value
    return curr

# Function to remove a node with the given value from the BST
def remove(root, val):
    # If the current node is None, return None
    if not root:
        return None
    
    # If the value to remove is greater than the current node's value,
    # go to the right subtree
    if val > root.val:
        root.right = remove(root.right, val)
    # If the value to remove is less than the current node's value,
    # go to the left subtree
    elif val < root.val:
        root.left = remove(root.left, val)
    # If the current node is the one to be removed
    else:
        # If the node has no left child, return the right child
        if not root.left:
            return root.right
        # If the node has no right child, return the left child
        elif not root.right:
            return root.left
        # If the node has both children
        else:
            # Find the inorder successor (smallest in the right subtree)
            minNode = minValueNode(root.right)
            # Copy the inorder successor's value to the current node
            root.val = minNode.val
            # Remove the inorder successor
            root.right = remove(root.right, minNode.val)
    # Return the root of the modified subtree
    return root

```

- **DFS O(n)**

```python
# Depth First Search Algorithm
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

#recursive solution
# store found nodes
nodes = []

 def inorder(root):
    # past leaf return nothing
    if not root:
        return
    # traverse left
    inorder(root.left)
    # when leaf found store value
    nodes.append(root.val)
    # traverse right
    inorder(root.right)

    inorder(root)
    return nodes

def preorder(root):
    if not root:
        return
    print(root.val)
    preorder(root.left)
    preorder(root.right)

def postorder(root):
    if not root:
        return
    postorder(root.left)
    postorder(root.right)
    print(root.val)
```

- **BFS O(n)**

```python

from collections import deque

class TreeNode:
    # TreeNode class definition
    def __init__(self, val):
        self.val = val       # Node value
        self.left = None     # Left child
        self.right = None    # Right child

def bfs(root):
    # Breadth-first search (BFS) function
    queue = deque()        # Initialize a deque to use as the queue

    if root:
        queue.append(root) # Start with the root node
    
    level = 0              # Initialize level counter
    while len(queue) > 0:  # Continue until the queue is empty
        print("level: ", level)
        for i in range(len(queue)):  # Iterate over nodes at the current level
            curr = queue.popleft()   # Pop the leftmost node
            print(curr.val)          # Print the value of the current node
            if curr.left:
                queue.append(curr.left)  # Add left child to the queue if it exists
            if curr.right:
                queue.append(curr.right) # Add right child to the queue if it exists
        level += 1  # Increment level counter after processing all nodes at the current level

```

### Graphs

- Set of vertices connected by edges
- Directed and undirected graphs

```python
# A simple representation of graph using adjacency list
graph = {
    'A' : ['B','C'],
    'B' : ['D', 'E'],
    'C' : ['F'],
    'D' : [],
    'E' : ['F'],
    'F' : []
}

# Function to add an edge to graph
def add_edge(graph, u, v):
    graph[u].append(v)

```

### Hash Tables

- Key-value storage
- Average constant-time complexity for operations

```python
# Creating a hashtable using dictionary
hash_table = {'name': 'Alice', 'age': 25, 'location': 'New York'}

# Accessing value
print(hash_table['name'])  # Output: Alice

# Adding a new key-value pair
hash_table['occupation'] = 'Engineer'

```

## Algorithms

### Sorting

- **Bubble Sort O(n^2)**

```python
# Bubble Sort implementation
def bubble_sort(arr):
    n = len(arr)
    # Traverse through all array elements
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n-i-1):
            # Traverse the array from 0 to n-i-1
            # Swap if the element found is greater than the next element
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
```

- **Merge Sort O(n log n)**

```python
        def mergeTwoSortedArrays(left, right, nums):
            # iterator for left arr, right arr, and full arr
            i, j, k = 0, 0, 0
            # loop until we reach the end of the shortest arr
            while i < len(left) and j < len(right):
                # insert left first
                if left[i] < right[j]:
                    nums[k] = left[i]
                    i+=1
                # insert right first
                else:
                    nums[k] = right[j]
                    j+=1
                # next index to place
                k+=1

            # append the rest left over to end of nums
            nums[k:] = left[i:] if i < len(left) else right[j:]


        def mergesort(nums):
            # base case signle el arr
            if len(nums) == 1: return

            # get middle index
            mid = len(nums)//2

            # get right half of arr
            left = nums[:mid]

            # get left half of arr
            right = nums[mid:]

            # sort left array
            mergesort(left)

            # sort right array
            mergesort(right)

            # merge the left and right sorted arrays together
            mergeTwoSortedArrays(left, right, nums)

        mergesort(nums)
        return nums
```

- **Quick Sort amoritized O(n log n) worst case O(n^2) if array already sorted**

```python
def quickSort(arr, start, end):
    if end - start + 1 <= 1:
        return

    pivot = arr[end]
    left = start # pointer for left side

    # Partition: elements smaller than pivot on left side
    for i in range(start, end):
        if arr[i] < pivot:
            tmp = arr[left]
            arr[left] = arr[i]
            arr[i] = tmp
            left += 1

    # Move pivot in-between left & right sides
    arr[end] = arr[left]
    arr[left] = pivot

    # Quick sort left side
    quickSort(arr, start, left - 1)

    # Quick sort right side
    quickSort(arr, left + 1, end)

    return arr
```

### Searching

- **Linear Search O(n)**
- **Binary Search O(log n)**

```python
# Binary Search implementation
def binary_search(arr, x):
    low = 0
    high = len(arr) - 1
    mid = 0

    while low <= high:
        mid = (high + low) // 2
        if arr[mid] < x:
            low = mid + 1
        elif arr[mid] > x:
            high = mid - 1
        else:
            return mid
    return -1
```

### Dynamic Programming

- Technique to solve problems by breaking them down into simpler subproblems

```python
# Fibonacci Series using Dynamic Programming
def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 2:
        return 1
    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    return memo[n]

```

### Backtracking

- Algorithmic technique for solving recursive problems by trying to build a solution incrementally

```python
# N-Queens problem using backtracking

def is_safe(board, row, col, n):
    # Check this row on left side for any queens (1s)
    for i in range(col):
        if board[row][i] == 1:
            return False  # A queen is already placed in this row

    # Check upper diagonal on left side for any queens
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False  # A queen is already placed in this upper diagonal

    # Check lower diagonal on left side for any queens
    for i, j in zip(range(row, n, 1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False  # A queen is already placed in this lower diagonal

    return True  # No queens are threatening this position

def solve_n_queens(board, col, n):
    # Base case: If all queens are placed
    if col >= n:
        return True

    # Consider this column and try placing a queen in all rows one by one
    for i in range(n):
        if is_safe(board, i, col, n):
            board[i][col] = 1  # Place a queen at (i, col)

            # Recur to place the rest of the queens
            if solve_n_queens(board, col + 1, n):
                return True  # If placing queen leads to a solution

            board[i][col] = 0  # Backtrack if placing queen doesn't lead to a solution

    return False  # If the queen cannot be placed in any row in this column


```

- **Bit Manipulation**

```python
# AND
n = 1 & 1

# OR
n = 1 | 0

# XOR
n = 0 ^ 1

# NOT (negation)
n = ~n

# Bit shifting
n = 1
n = n << 1
n = n >> 1

# Counting Bits
def countBits(n):
    count = 0
    while n > 0:
        if n & 1 == 1: # Check right bit is 1
            count += 1
        n = n >> 1 # same as n // 2. Shift bit to right
    return count

```

## Problem Solving Patterns

- **Sliding Window**

```python
def max_sum_subarray(arr, k):
    # Check if the length of the array is less than k
    if len(arr) < k:
        print("Invalid")
        return -1

    # Compute the sum of the first window of size k
    window_sum = sum([arr[i] for i in range(k)])
    max_sum = window_sum

    # Slide the window from start to end in the array
    for i in range(len(arr) - k):
        # Slide the window by one element to the right
        window_sum = window_sum - arr[i] + arr[i + k]
        # Update the maximum sum if needed
        max_sum = max(max_sum, window_sum)

    return max_sum

# Example usage:
arr = [1, 2, 5, 2, 8, 1, 5]
k = 3
print("Maximum sum of a subarray of size k:", max_sum_subarray(arr, k))
```

- **Two Pointers**

Problem: Given a sorted array of integers and a target value, find if there’s a pair of numbers that add up to the target.

Solution:

```python
def two_pointer_approach(nums, target):
    left, right = 0, len(nums) - 1  # Initialize two pointers

    while left < right:  # Continue until they meet
        current_sum = nums[left] + nums[right]

        if current_sum == target:  # Check if the current sum matches the target
            return True  # We found a pair
        elif current_sum < target:  # If current sum is less, move the left pointer to the right
            left += 1
        else:  # If current sum is more, move the right pointer to the left
            right -= 1

    return False  # No pair found that adds up to the target

# Example usage:
nums = [1, 2, 3, 4, 5, 6]
target = 9
print(two_pointer_approach(nums, target))  # Output: True, because 3 + 6 = 9

```
