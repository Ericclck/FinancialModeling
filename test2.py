import random

def move_A(x, y):
    """Move A either up or right."""
    if x == 2:  # On the rightmost column, can only move up
        return x, y + 1
    elif y == 2:  # On the top row, can only move right
        return x + 1, y
    else:
        return (x + 1, y) if random.choice([True, False]) else (x, y + 1)

def move_B(x, y):
    """Move B either down or left."""
    if x == 0:  # On the leftmost column, can only move down
        return x, y - 1
    elif y == 0:  # On the bottom row, can only move left
        return x - 1, y
    else:
        return (x - 1, y) if random.choice([True, False]) else (x, y - 1)

def simulate_meeting():
    """Simulate the paths of A and B to check if they meet."""
    # Starting positions
    A = (0, 0)
    B = (2, 2)
    
    while not ((A[0] == 2 and A[1] == 2) and (B[0] == 0 and B[1] == 0)):
        if A == B:  # They've met
            return True
        A = move_A(*A)
        B = move_B(*B)
    return False

# Monte Carlo Simulation
num_simulations = 100000
meet_count = sum(simulate_meeting() for _ in range(num_simulations))

probability_meet_monte_carlo = meet_count / num_simulations
print(probability_meet_monte_carlo)