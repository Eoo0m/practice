from __future__ import annotations
from dataclasses import dataclass
from typing import TypeVar, Generic, List, Optional


@dataclass
class Edge:
  u: int
  v: int

  def __str__(self) -> str:
    return f"{self.u} -> {self.v}"

  def reversed(self) -> Edge:
    return Edge(self.v, self.u)


V = TypeVar('V')
class Graph(Generic[V]):
  def __init__(self, vertices: List = []) -> None:
    self.vertices: List[V] = vertices
    self.edges: List[List[Edge]] = [[] for _ in vertices]

  @property
  def vertex_count(self) -> int:
    return len(self.vertices)

  @property
  def edge_count(self) -> int:
    return sum(map(len, self.edges))

  def add_vertex(self, vertex: V) -> None:
    self.vertices.append(vertex)
    self.edges.append([])
    return self.vertex_count-1

  def add_edge(self, edge: Edge) -> None:
    self.edges[edge.u].append(edge)
    self.edges[edge.v].append(edge.reversed())


  def add_edge_by_indices(self, u: int, v: int) -> None:
    self.edges[u].append(Edge(u, v))
    self.edges[v].append(Edge(v, u))

  def add_edge_by_vertices(self, first:V, second:V) -> None:
    self.add_edge_by_indices(self.vertices.index(first), self.vertices.index(second))

  def vertex_at(self, index:int) -> V:
    return self.vertices[index]

  def index_of(self, vertex:V) -> int:
    return self.vertices.index(vertex)

  def neighbors_for_index(self, index:int) -> List[V]:
    return list(map(self.vertex_at, [e.v for e in self.edges[index]]))
    


