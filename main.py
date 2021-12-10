import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def create_stiffness_matrix(nodes, elements, properties, dof):

    # DEFINE EMPTY GLOBAL STIFFNESS MATRIX
    k = np.zeros([dof, dof])

    # POPULATE GLOBAL STIFFNESS MATRIX
    for element in elements:
        x1 = nodes[element[0]][0]
        x2 = nodes[element[1]][0]
        y1 = nodes[element[0]][1]
        y2 = nodes[element[1]][1]

        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        c = (x2 - x1) / length
        s = (y2 - y1) / length

        # CREATE ELEMENT STIFFNESS MATRIX
        k_el = properties[element[2]][0] * properties[element[2]][1] / length * np.array([
            [c ** 2, c * s, -c ** 2, -c * s],
            [c * s, s ** 2, -c * s, -s ** 2],
            [-c ** 2, -c * s, c ** 2, c * s],
            [-c * s, -s ** 2, c * s, s ** 2],
        ])

        # COMBINE ELEMENT MATRIX INTO GLOBAL MATRIX
        k[element[0] * 2: element[0] * 2 + 2, element[0] * 2: element[0] * 2 + 2] += k_el[:2, :2]
        k[element[0] * 2: element[0] * 2 + 2, element[1] * 2: element[1] * 2 + 2] += k_el[:2, 2:]
        k[element[1] * 2: element[1] * 2 + 2, element[0] * 2: element[0] * 2 + 2] += k_el[2:, :2]
        k[element[1] * 2: element[1] * 2 + 2, element[1] * 2: element[1] * 2 + 2] += k_el[2:, 2:]

    return k


def solve_fea(k_matrix, f_matrix, dof):

    # REMOVE NONE VALUES FROM F TO ALLOW SOLVING
    reduction_index = []
    for i, value in enumerate(f_matrix):
        if not np.isnan(value):
            reduction_index.append(i)
    f_reduced = f_matrix[reduction_index]
    k_reduced = k_matrix[reduction_index, :][:, reduction_index]

    # CALCULATE FOR DISPLACEMENTS
    u_reduced = np.linalg.solve(k_reduced, f_reduced)
    u_matrix = np.zeros(dof)
    u_matrix[reduction_index] = u_reduced

    # CALCULATE REACTION FORCES
    f_matrix = np.matmul(k_matrix, u_matrix)

    return u_matrix, f_matrix


def create_force_matrix(forces, constraints, dof):

    # CREATE GLOBAL FORCE MATRIX
    f = np.zeros(dof)
    for force in forces:
        f[force[0] * 2 + force[1]] += force[2]
    for constraint in constraints:
        f[constraint[0] * 2 + constraint[1]] = None

    return f


def calculate_forces(nodes, elements, properties, u_matrix):

    # DEFINE EMPTY MATRICES TO POPULATE WITH VALUES
    strain = np.zeros(len(elements))
    stress = np.zeros(len(elements))
    force = np.zeros(len(elements))

    # CALCULATE FORCE, STRESS AND STRAIN FOR EACH ELEMENT
    for index, element in enumerate(elements):

        # CALCULATE DESIGN LENGTH OF MEMBER
        x1 = nodes[element[0]][0]
        x2 = nodes[element[1]][0]
        y1 = nodes[element[0]][1]
        y2 = nodes[element[1]][1]
        length_design = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

        # CALCULATE DEFLECTED LENGTH OF MEMBER
        x1 = nodes[element[0]][0] + u_matrix[element[0] * 2 + 0]
        x2 = nodes[element[1]][0] + u_matrix[element[1] * 2 + 0]
        y1 = nodes[element[0]][1] + u_matrix[element[0] * 2 + 1]
        y2 = nodes[element[1]][1] + u_matrix[element[1] * 2 + 1]
        length_deflected = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

        # CALCULATE ENGINEERING STRAIN, ENGINEERING STRESS AND INTERNAL FORCE OF MEMBER
        strain[index] = (length_deflected - length_design) / length_design
        stress[index] = strain[index] * properties[element[2]][0]
        force[index] = stress[index] * properties[element[2]][1]

    return strain, stress, force


def plot_results(nodes, elements, u_matrix, colour_scale=None, deflection_factor=1):

    # PLOT DESIGN POSITIONS
    for element in elements:
        x1 = nodes[element[0]][0]
        x2 = nodes[element[1]][0]
        y1 = nodes[element[0]][1]
        y2 = nodes[element[1]][1]
        plt.plot([x1, x2], [y1, y2], color='grey', linestyle='dashed')

    # PLOT DEFLECTED POSITIONS
    for index, element in enumerate(elements):
        x1 = nodes[element[0]][0] + u_matrix[element[0] * 2 + 0] * deflection_factor
        x2 = nodes[element[1]][0] + u_matrix[element[1] * 2 + 0] * deflection_factor
        y1 = nodes[element[0]][1] + u_matrix[element[0] * 2 + 1] * deflection_factor
        y2 = nodes[element[1]][1] + u_matrix[element[1] * 2 + 1] * deflection_factor
        if colour_scale is None:
            plt.plot([x1, x2], [y1, y2], color='r')
        else:
            plt.plot([x1, x2], [y1, y2], color=cm.bwr(colour_scale[index]))

    # SHOW PLOT
    plt.show()


def main():

    # DEFINE NODE LOCATIONS
    # [x dimension, y dimension]
    nodes = [
        [0, 0],
        [10, 0],
        [20, 0],
        [30, 0],
        [40, 0],
        [5, 5],
        [15, 5],
        [25, 5],
        [35, 5]
    ]

    # DEFINE PROPERTIES
    # (elastic modulus, area)
    properties = [
        (210 * 10 ** 6, 0.1 * 0.1)
    ]

    # DEFINE BEAM ELEMENTS
    # (node1 id, node2 id, property_id)
    elements = [
        (0, 1, 0),
        (1, 2, 0),
        (2, 3, 0),
        (3, 4, 0),
        (0, 5, 0),
        (1, 5, 0),
        (1, 6, 0),
        (2, 6, 0),
        (2, 7, 0),
        (3, 7, 0),
        (3, 8, 0),
        (4, 8, 0),
        (5, 6, 0),
        (6, 7, 0),
        (7, 8, 0)
    ]

    # DEFINE CONSTRAINTS
    # (node id, direction)
    constraints = [
        (0, 0),
        (0, 1),
        (4, 1)
    ]

    # DEFINE APPLIED POINT FORCES
    # (node id, direction, magnitude)
    forces = [
        (6, 1, -10000),
        (7, 1, -10000)
    ]

    # CALCULATE DEGREES OF FREEDOM IN SYSTEM
    dof = len(nodes) * 2

    # CREATE GLOBAL STIFFNESS MATRIX
    k_matrix = create_stiffness_matrix(nodes, elements, properties, dof)

    # CREATE FORCES MATRIX
    f_matrix = create_force_matrix(forces, constraints, dof)

    #  SOLVE FEA EQUATIONS TO FIND DEFLECTIONS AND REACTIONS MATRICES
    u_matrix, f_matrix = solve_fea(k_matrix, f_matrix, dof)

    # FIND FORCES AND STRESS IN ELEMENTS
    strain, stress, force = calculate_forces(nodes, elements, properties, u_matrix)

    # PLOT RESULTS
    plot_results(nodes, elements, u_matrix, colour_scale=force)


if __name__ == '__main__':
    main()
