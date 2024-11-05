from random import choice
from string import ascii_uppercase

class CSP:
  def __init__(self, variables, domains):
    self.variables = variables
    self.domains = domains
    self.constraints = {}
    for variable in self.variables:
      self.constraints[variable] = []


  def add_constraint(self, constraint):
    for variable in constraint.variable:
      if variable in self.variables:
        self.constraints[variable].append(constraint)

  
  def backtracking_search(self, assignment= {}):

    if len(assignment) == len(self.variables):
      return assignment

    unassigned = [v for v in self.variables if v not in assignment]
    first_element = unassigned[0]
    print("assignment",assignment, "unassigned", unassigned)

    for value in self.domains[first_element]:
      local_assignment = assignment.copy()
      local_assignment[first_element] = value
      if self.consistent(first_element, local_assignment ): 
        result = self.backtracking_search(local_assignment)
        if result is not None:
          return result

    return None

  def consistent(self, variable, assignment):
    for constraint in self.constraints[variable]:
      if not constraint.satisfied(assignment):
        return False
    
    return True






class constraint:
  def __init__(self, words):
    self.variable = words

  def satisfied(self, assignment):
    locations = [locs for value in assignment.values() for locs in value]
    print(locations)
    return len(locations) == len(set(locations))


def generate_grid(rows, columns):
  return[[choice(ascii_uppercase) for c in range(columns)] for r in range(rows)]

def display_grid(grid):
  for row in grid:
    print("".join(row))


def generate_domain(word, grid):
  domains = []
  height = len(grid)
  width = len(grid[0])
  length = len(word)

  for row in range(height):
    for col in range(width):
      if col + length <= width:
        domains.append([(row, col+i) for i in range(length)])
        if row + length <= height:
          domains.append([(row+i,col-i) for i in range(length)])

      if row + length <= height:
        domains.append([(row+i, col) for i in range(length)])
        if col+1 - length >= 0:
          domains.append([(row-i,col+i) for i in range(length)])


  return domains


grid = generate_grid(4,4)
display_grid(grid)
variables = ["MAX", "JOE"]
domains = {}
for word in variables:
  domains[word] = generate_domain(word, grid)

print("domains",domains)

csp = CSP(variables, domains)
csp.add_constraint(constraint(variables))
solution = csp.backtracking_search()

print(solution)
