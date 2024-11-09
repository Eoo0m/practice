from typing import NamedTuple, List, Dict, Optional, Tuple
from random import choice
from string import ascii_uppercase
from typing import Generic, TypeVar, Dict, List, Optional


class GridLocation(NamedTuple):
    row: int
    column: int


class Constraint:
    def __init__(self, circuits):
        self.variables = circuits

    def satisfied(self, assignment) :
        all_locations = []

        for location in assignment.values():
            for row in range(location[0].row, location[1].row+1):
                for col in range(location[0].column, location[1].column+1):
                    all_locations.append(GridLocation(row,col))

        return len(set(all_locations)) == len(all_locations)

class CSP():
    def __init__(self, variables, domains) :
        self.variables = variables
        self.domains = domains
        self.constraints = {}

        for variable in self.variables:
            self.constraints[variable] = []
            if variable not in self.domains:
                raise LookupError("모든 변수에 도메인이 할당되어야 합니다.")

    def add_constraint(self, constraint):
        for variable in constraint.variables:
            if variable not in self.variables:
                raise LookupError("제약 조건 변수가 아닙니다.")
            else:
                self.constraints[variable].append(constraint)

    def consistent(self, variable, assignment):
        for constraint in self.constraints[variable]:
            if not constraint.satisfied(assignment):
                return False
        return True



    def backtracking_search(self, assignment = {}) :
        if len(assignment) == len(self.variables):
            return assignment

        unassigned= [v for v in self.variables if v not in assignment]

        first: V = unassigned[0]
        for value in self.domains[first]:
            local_assignment = assignment.copy()
            # local_assignment = assignment
            local_assignment[first] = value
            if self.consistent(first, local_assignment):
                result: Optional[Dict[V, D]] = self.backtracking_search(local_assignment)
                if result is not None:
                    return result
        return None




def generate_grid(rows, columns):
    return [["0" for c in range(columns)] for r in range(rows)]


def display_grid(grid) :
    for row in grid:
        print("".join(row))


def generate_domain(circuit_size, grid):
    domain = []
    height= len(grid)
    width= len(grid[0])

    circuit_height = circuit_size[0]-1
    circuit_width = circuit_size[1]-1

    for row in range(height):
        for col in range(width):
          if col + circuit_width <= width and row + circuit_height <= height:
            domain.append((GridLocation(row, col),GridLocation(row+circuit_height, col+circuit_width)))

    return domain



if __name__ == "__main__":
    grid= generate_grid(9, 9)
    #"circuit name": (circuits'size)
    circuits = {"1":(2,7), "2":(1,8), "3":(6,1), "4":(3,3)}
    locations = {}
    for circuit, circuit_size in circuits.items():
        locations[circuit] = generate_domain(circuit_size, grid)
        print(locations)
    csp = CSP(circuits, locations)
    csp.add_constraint(Constraint(circuits))
    solution = csp.backtracking_search()

    print(solution)
    for circuit, size in solution.items():
      for row in range(size[0].row, size[1].row+1):
        for col in range(size[0].column, size[1].column+1):
          grid[row][col] = circuit
    display_grid(grid)
