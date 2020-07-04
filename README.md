# DS_Assignment for Cognizant Data Science Summit 2020
Python program for Eight Queens puzzle with genetic programming.

How to Run:

$python 8_queens_Arpan_5.py

Note:
The starter API is "eight_queens_puzzle" which takes the arument as number of queens.
#------------------------------------------------------------------------------#
nq = 8 ## Number of Queens -- hard-coded

result = eight_queens_puzzle(nq)

## to check the final result
for row in result:
    print (" ".join(row))
#------------------------------------------------------------------------------#

Final output:

Maximum Fitness = 28
Solved in Generation 15457!

One of the solutions: 
Chromosome = [5, 1, 8, 6, 3, 7, 2, 4],  Fitness = 28
x x Q x x x x x
x x x x x Q x x
x x x Q x x x x
Q x x x x x x x
x x x x x x x Q
x x x x Q x x x
x x x x x x Q x
x Q x x x x x x


