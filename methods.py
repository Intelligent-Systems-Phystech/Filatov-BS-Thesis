import torch
import numpy as np

class MinNormSolver():
    """
    Solver for Minimum simplex norm problem
    """
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2) )
        cost = v2v2 + gamma*(v1v2 - v2v2)
        return gamma, cost

    def _min_norm_2d(vecs, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        for i in range(len(vecs)):
            for j in range(i+1,len(vecs)):
                if (i,j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i,j)] += torch.mul(vecs[i][k], vecs[j][k]).sum().data
                    dps[(j, i)] = dps[(i, j)]
                if (i,i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i,i)] += torch.mul(vecs[i][k], vecs[i][k]).sum().data
                if (j,j) not in dps:
                    dps[(j, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(j, j)] += torch.mul(vecs[j][k], vecs[j][k]).sum().data
                c,d = MinNormSolver._min_norm_element_from2(dps[(i,i)], dps[(i,j)], dps[(j,j)])
                if d < dmin:
                    dmin = d
                    sol = [(i,j),c,d]
        return sol, dps

    def find_min_norm_element_FW(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the Frank Wolfe until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n=len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec , init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                grad_mat[i,j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))

            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]

            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc*sol_vec
            new_sol_vec[t_iter] += 1 - nc

            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec


def gradient_normalization(normalization_type, grads):
    """

    Args:
        normalization_type: l1 or l2
        grads: tuple of task grddients

    Returns:

    """
    gn = {}
    grads = grads.copy()
    if normalization_type == 'l2':
        for i in range(len(grads)):
            gn[i] = 0.0
            for j in range(len(grads[i])):
                gn[i] += grads[i][j].pow(2).sum()

            grads[i] = tuple(map(lambda x: x / gn[i], grads[i]))

    return grads, gn


def change_gradient(method, grads):
    """

    Args:
        method: MGDA or EDM
        grads: tuple of task gradients

    Returns:

    """
    # grads1, grads2 = grads
    # g1 = torch.nn.utils.parameters_to_vector(grads1)
    # g2 = torch.nn.utils.parameters_to_vector(grads2)

    # wandb.log({'g1 norm': torch.norm(g1),
    #            'g2 norm': torch.norm(g2),
    #            'cos'  : torch.cosine_similarity(g1, g2, dim=0)})

    # if method == 'MGDA':
    #     coefs = altitude_direction(grads)

    # if method == 'PC':
    #     g = pcgrad_direction(g1, g2)
    # elif 'NORM' in method:
    #     NORM = int(method.split('_')[1])
    #     g = norm_direction(g1, g2, NORM)
    # elif method == 'EDM':
    #     g = bisection_direction(g1, g2)
    if method == "MGDA":
        prom_grads = grads

    if method == "EDM":
        prom_grads, _ = gradient_normalization("l2", grads)

    sol, _ = MinNormSolver.find_min_norm_element_FW(prom_grads)
    g = tuple(map(lambda x: x * sol[0], grads[0]))
    for i in range(1, len(grads)):
        prom = tuple(map(lambda x: x * sol[i], grads[i]))
        g = tuple(map(lambda x, y: x + y, g, prom))

    # wandb.log({'h norm': torch.norm(g)})
    # torch.nn.utils.vector_to_parameters(g, grads)
    return g