import terminus.ga201.join as join
import numpy as np
from itertools import combinations

class ConvexBody:
    def __init__(self, planes, inverted = False):
        self.planes = planes
        self.linear_formes_by_grades = [0] * (self.max_grade()) 
        self.find_linear_formes()
        self.inverted = inverted

    def max_grade(self):
        return 2

    def meet_of_hyperplanes_combination(self, planes):
        result = planes[0]
        for i in range(1, len(planes)):
            result = join.meet(result, planes[i])
        return result

    def internal_vertices(self, vertices):
        int_vertices = []
        for vertex in vertices:
            is_internal = True
            for plane in self.planes:
                is_internal = self.is_internal_point(vertex)
                if not is_internal:
                    break
            if is_internal:
                int_vertices.append(vertex)        
        return int_vertices

    def drop_infinite_points(self, vertices):
        non_infinite_points = []
        for vertex in vertices:
            if vertex.is_infinite():
                continue
            non_infinite_points.append(vertex)
        return non_infinite_points

    def meet_of_hyperplanes(self):
        # get list of all vertices of convex body
        # by list of planes
        # planes - list of planes

        # get list of all combination of planes by max_grade elements
        cmbs = [c for c in combinations(self.planes, self.max_grade())]
        vertices = []

        for cmb in cmbs:
            # get all vertices of cmb
            # and add them to vertices list
            vertices.append(self.meet_of_hyperplanes_combination(cmb))

        non_infinite_points = self.drop_infinite_points(vertices)
        print("non_infinite_points", non_infinite_points)
        int_vertices = self.internal_vertices(non_infinite_points)

        return int_vertices

    def find_linear_formes(self):
        # get list of all combination of planes by max_grade elements
        self.linear_formes_by_grades[self.max_grade() - 1] = self.planes
        vertices = self.meet_of_hyperplanes()
        self.linear_formes_by_grades[0] = vertices

        print(self.linear_formes_by_grades)
        
        #TODO: middle grades

    def count_of_vertices(self):
        return len(self.linear_formes_by_grades[0])

    def count_of_hyperplanes(self):
        return len(self.planes)
        
    def vertices(self):
        return self.linear_formes_by_grades[0]

    def hyperplanes(self):
        return self.linear_formes_by_grades[self.max_grade() - 1]

    def is_internal_point(self, point):
        for plane in self.planes:
            if join.oriented_distance(point, plane).to_float() > 1e-8:
                return False
        return True
    
    def point_projection(self, point):
        candidates = []
        for grade in range(self.max_grade()-1, -1, -1):
            print("grade", grade)
            for linear_form in self.linear_formes_by_grades[grade]:
                proj = join.point_projection(point, linear_form)
                print("proj", proj)
                if self.is_internal_point(proj):
                    candidates.append(proj)

            if len(candidates) == 0:
                continue

            print("point", point)
            print("candidates", candidates)
            distances = [join.distance_point_point(point, candidate).to_float() for candidate in candidates]
            print("distances", distances)
            min_distance_index = np.argmin(distances)
            return candidates[min_distance_index]

        #unreachable
        raise Exception("point_projection: unreachable")


