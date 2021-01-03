from gurobipy import *
import numpy
import time

# Number of values in a reaction graph
def ReactionNumber(A):
    number = 0
    for j in range(A.shape[1]):
        for k in range(A.shape[2]):
            if j != k:
                if A[0][j][k] > 0:
                    number += 1
    for i in range(1,A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(A.shape[2]):
                if A[i][j][k] > 0:
                    number += 1
    return number

# Value of a binary array
def Dec(bitlist):
    out = 0
    for bit in bitlist:
        out = (out << 1) | int(bit)
    return out

# Binarize a reaction graph
def Binarize(A, Enc):
    U = [0 for e in range(len(Enc))]
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(A.shape[2]):
                if (i != 0 or j != k) and A[i][j][k] > 0 and FindRow(Enc, numpy.array([i, j, k])) < Enc.shape[0]:
                    U[FindRow(numpy.array(Enc), numpy.array([i, j, k]))] = 1
    return U

# Find a complex y in the complex set Y
def FindComplex(Y, y):
    if Y.shape[1] == 0:
        return 1
    for q in range(Y.shape[1]):
        if numpy.array_equal([Y[:,q]], y):
            return q
    return Y.shape[1]

def FindRow(E, e):
    if E.shape[0] == 0:
        return 1
    for q in range(E.shape[0]):
        if numpy.array_equal(E[q,:], e):
            return q
    return E.shape[0]

# Find an appropriate complex set for realizing a delayed kinetic system (Y,M0,...,Mp)
def GenComp(Mp, Yp, p, m, n):
    Y = numpy.empty((n, 0))
    for i in range(1, p+1):
        for j in range(n):
            for k in range(m):
                if Mp[i][j][k] > 0:
                    if FindComplex(Y, numpy.eye(1, n, j)) >= Y.shape[1]:
                        Y = numpy.append(Y, numpy.transpose(numpy.eye(1, n, j)), axis = 1)
                    for l in range(n):
                        if Yp[l][k] > 0:
                            Mp[0][l][k] += Yp[l][k] * Mp[i][j][k]
    for j in range(n):
        for k in range(m):
            if Mp[0][j][k] != 0:
                if FindComplex(Y, Yp[:,k]+numpy.sign(Mp[0][j][k])*numpy.eye(1, n, j)) >= Y.shape[1]:
                    Y = numpy.append(Y, numpy.transpose(Yp[:,k]+numpy.sign(Mp[0][j][k])*numpy.eye(1, n, j)), axis = 1)
    for k in range(m):
        if FindComplex(Y, Yp[:,k] + numpy.zeros((1, n))) >= Y.shape[1]:
            Y = numpy.append(Y, numpy.transpose([Yp[:,k]]), axis = 1)
    M = numpy.zeros((p+1, n, Y.shape[1]))
    for j in range(Y.shape[1]):
        for k in range(Yp.shape[1]):
            if numpy.array_equal(Y[:,j], Yp[:,k]):
                for l in range(p+1):
                    M[l,:,j] = Mp[l,:,k]
                continue
    return Y, M

# Find the dynamically equivalent canonical realization of the delayed kinetic system (Y',M0,...,Mp)
def DDE2CRN(Yp, Mp, p, m, n):
    Mpp = numpy.copy(Mp)
    Y, M = GenComp(Mp, Yp, p, m, n)
    m = M.shape[2]
    A = numpy.zeros(((p+1, m, m)))
    for i in range(1,p+1):
        for j in range(n):
            for k in range(m):
                if M[i][j][k] > 0:
                    q = FindComplex(Y, Y[:,k]+numpy.zeros((1,n)))
                    r = FindComplex(Y, numpy.eye(1, n, j))
                    A[i][r][q] += M[i][j][k]
                    A[0][q][q] -= M[i][j][k]
    for j in range(n):
        for k in range(m):
            if M[0][j][k] != 0:
                q = FindComplex(Y, Y[:,k]+numpy.zeros((1,n)))
                r = FindComplex(Y, Y[:,k]+numpy.sign(M[0][j][k])*numpy.eye(1, n, j))
                A[0][r][q] += abs(M[0][j][k])
                A[0][q][q] -= abs(M[0][j][k])
    M = numpy.zeros((p+1, n, Y.shape[1]))
    for j in range(Y.shape[1]):
        for k in range(Yp.shape[1]):
            if numpy.array_equal(Y[:,j], Yp[:,k]):
                for l in range(p+1):
                    M[l,:,j] = Mpp[l,:,k]
                continue
    return Y, A, M

def ConvComb(Results):
    return sum(Results) / len(Results)

def Decomp(Result, L, Y, p, m, n):
    # Calculate the matrix invT
    T = numpy.zeros((n,n))
    for i in range(n):
        T[i][i] = 1/Result[i]

    # Calculate the matrices M
    M = numpy.zeros((p+1,n,m))
    for i in range(p+1):
        for j in range(n):
            for k in range(m):
                M[i][j][k] = Result[n+i*n*m+j*m+k]

    # Calculate the matrices A
    A = numpy.zeros((p+1,m,m))
    for j in range(m):
        for k in range(j):
            A[0][j][k] = Result[n+(p+1)*n*m+k*(m-1)+j-1]
        for k in range(j+1,m):
            A[0][j][k] = Result[n+(p+1)*n*m+k*(m-1)+j]
    for i in range(1,p+1):
        for j in range(m):
            for k in range(m):
                A[i][j][k] = Result[n+(p+1)*n*m+i*m*m+(k-1)*m+j]

    if False:
        # No 0 -> 0 reactions
        q = FindComplex(Y,[0 for i in range(n)]+numpy.zeros((1,n)))
        if q < Y.shape[1]:
            for i in range(p+1):
                A[i][q][q] = 0

            # No delayed C -> 0 reactions
            for i in range(1,p+1):
                for k in range(m):
                    A[0][q][k] += A[i][q][k]
                    A[i][q][k] = 0

            # Repair L inequalities
            for i in range(len(L)):
                if numpy.array_equal(L[i][0], numpy.zeros(n)) and sum(L[i][1]) == 1 and L[i][2] == 0:
                    for k in range(q):
                        if L[i][1][k*(m-1)+q-1] > 0:
                            A[0][q][k] = 0
                    for k in range(q+1,m):
                        if L[i][1][k*(m-1)+q] > 0:
                            A[0][q][k] = 0
    for j in range(m):
        A[0][j][j] = -quicksum(quicksum(A[i][k][j] for k in range(m)) for i in range(p+1)).getValue()

    psiT = numpy.eye(m)
    for i in range(m):
        for j in range(n):
            psiT[i][i] *= T[j][j]**Y[j][i]

#    return numpy.linalg.inv(T), numpy.matmul(T,M), numpy.matmul(A,psiT)
    return numpy.linalg.inv(T), numpy.matmul(T,M), A

def M2P(M, Y, p, m, n):
    P = []
    for i in range(p+1):
        for j in range(n):
            for k in range(m):
                P.append([numpy.reshape(numpy.eye(1,(p+1)*n*m,i*n*m+j*m+k),(p+1)*n*m),M[i][j][k]])
                P.append([-numpy.reshape(numpy.eye(1,(p+1)*n*m,i*n*m+j*m+k),(p+1)*n*m),-M[i][j][k]])
    return P

# Find a linearly conjugate realization of the delayed kinetic system (Y',M0,...,Mp)
def FindPositive(P, L, Y, p, m, n, H, dyn_equiv):

    # Create a new model
    model = Model("linearly conjugate realization")

    # Add variables
    q = model.addVars(n+(p+1)*n*m+(p+1)*m*m-m, lb = -100000.0, ub = 100000.0, vtype = GRB.CONTINUOUS, name = "q")

    # Set objective function
    model.setObjective(quicksum(q[j] for j in H), GRB.MAXIMIZE)

    # Set constraints
    model.addConstrs((q[j] <= 10 for j in range(n)), name = "invT bound")
    model.addConstrs((q[j] <= 100000 for j in range(n+(p+1)*n*m,n+(p+1)*n*m+(p+1)*m*m-m)), name = "Ai bound")
    model.addConstrs((q[j] >= 0 for j in range(n)), name = "invT positive")
    model.addConstrs((q[j] >= 0 for j in range(n+(p+1)*n*m,n+(p+1)*n*m+(p+1)*m*m-m)), name = "Ai positive")
    model.addConstrs((q[j] >= 0 for j in range(n+n*m,n+(p+1)*n*m)), name = "Mi positive")
    model.addConstrs((q[j] >= 0 for j in range(n,n+n*m) if Y[int(numpy.floor((j-n)/m))][(j-n)%m] == 0), name = "M0 kinetic")

    if dyn_equiv:
        model.addConstrs((q[j] == 1.0 for j in range(n)), name = "dynamically equivalent")
    z = FindComplex(Y,[0 for i in range(n)]+numpy.zeros((1,n)))
    if z < Y.shape[1]:
        model.addConstrs((q[n+(p+1)*n*m+i*m*m+(z-1)*m+z] <= 0 for i in range(1,p+1)), name = "no 0->0")
        model.addConstrs((q[n+(p+1)*n*m+i*m*m+(k-1)*m+z] <= 0 for i in range(1,p+1) for k in range(m)), name = "no C->0")

    model.addConstrs((quicksum(P[i][0][j]*q[n+j] for j in range((p+1)*n*m)) <= P[i][1] for i in range(len(P))), name = "P inequality")
    if L != []:
        model.addConstrs((quicksum(L[i][0][j]*q[j] for j in range(n))+quicksum(L[i][1][j]*q[n+(p+1)*n*m+j] for j in range((p+1)*m*m-m)) <= L[i][2] for i in range(len(L))), name = "L inequality")
    model.addConstrs((q[n+j*m+k]-quicksum((Y[j][l]-Y[j][k])*q[n+(p+1)*n*m+k*(m-1)+l] for l in range(k))-quicksum((Y[j][l]-Y[j][k])*q[n+(p+1)*n*m+k*(m-1)+l-1] for l in range(k+1,m))+Y[j][k]*quicksum(quicksum(q[n+(p+1)*n*m+i*m*m+(k-1)*m+l] for i in range(1,p+1)) for l in range(m)) == 0 for j in range(n) for k in range(m)), name = "M0=YA0")
    model.addConstrs((q[n+i*n*m+j*m+k]-quicksum(Y[j][l]*q[n+(p+1)*n*m+i*m*m+(k-1)*m+l] for l in range(m)) == 0 for i in range(1,p+1) for j in range(n) for k in range(m)), name = "Mi=YAi")

#        model.addConstrs((quicksum(L[i][j]*q[j] for j in range(n)) <= L[i][len(L[i])-1] for i in range(len(L))), name = "L inequality")
#    model.addConstrs((M[0][j][k]*q[j]-quicksum((Y[j][l]-Y[j][k])*q[k*(m-1)+n+l] for l in range(k))-quicksum((Y[j][l]-Y[j][k])*q[k*(m-1)+n+l-1] for l in range(k+1,m))+Y[j][k]*quicksum(quicksum(q[i*m*m+(k-1)*m+n+l] for i in range(1,p+1)) for l in range(m)) == 0 for j in range(n) for k in range(m)), name = "inv(T)*M0=YA0")
#    model.addConstrs((M[i][j][k]*q[j]-quicksum(Y[j][l]*q[i*m*m+(k-1)*m+n+l] for l in range(m)) == 0 for i in range(1,p+1) for j in range(n) for k in range(m)), name = "inv(T)*Mi=YAi")

    # Get the optimized model
    model.Params.OutputFlag = 0
    model.optimize()

    if model.status == 3:
        return numpy.array([0 for j in range(len(q))]), numpy.array([])

    #Calculate the point Q
    qValues = numpy.array([q[j].x for j in range(len(q))])

    #Calculate the index set B
    B = numpy.array([j for j in H if qValues[j] > 0])

    return qValues, B

# Find a dense linearly conjugate realization of the delayed kinetic system (Y',M0,...,Mp)
def DenseRealization(P, L, Y, p, m, n, dyn_equiv):
    # Initial index sets
    H = [j for j in range(n+(p+1)*n*m,n+(p+1)*n*m+(p+1)*m*m-m)]
    B = [j for j in range(n)] + H

    # Initialize Result and loops
    Results = []

    while len([j for j in B if j >= n]) > 0:
        H = [j for j in range(n)] + H
        Q, B = FindPositive(P, L, Y, p, m, n, H, dyn_equiv)
        Results = Results + [Q]
        H = [j for j in H if j not in B]

    if len([j for j in range(n) if j in H]) > 0:
        print("There is no linearly conjugate realization of the system.")
        return numpy.zeros((n, n)), numpy.zeros((p+1, n, m)), numpy.zeros((p+1, m, m))

    return Decomp(ConvComb(numpy.array(Results)), L, Y, p, m, n)

# Find a dense linearly conjugate realization of the delayed kinetic system (Y',M0,...,Mp)
def DenseRealizationM(Mp, L, Yp, p, m, n, generate, dyn_equiv):
    Mpp = numpy.copy(Mp)
    if generate:
        Y, Mtmp = GenComp(Mp, Yp, p, m, n)
        M = numpy.zeros((p+1, n, Y.shape[1]))
        for j in range(Y.shape[1]):
            for k in range(Yp.shape[1]):
                if numpy.array_equal(Y[:,j], Yp[:,k]):
                    for l in range(p+1):
                        M[l,:,j] = Mpp[l,:,k]
    else:
        Y = Yp
        M = Mpp
    m = M.shape[2]

    #Generate P from M
    P = M2P(M, Y, p, m, n)

    return Y, DenseRealization(P, L, Y, p, m, n, dyn_equiv)

# Compute a,b,d necessary for expressing that A[i][j][k]=0
def ZeroEdge(p, m, n, i, j, k):
    if i == 0:
        if j < k:
            b = numpy.reshape(numpy.eye(1,(p+1)*m*m-m,k*(m-1)+j),(p+1)*m*m-m)
        else:
            b = numpy.reshape(numpy.eye(1,(p+1)*m*m-m,k*(m-1)+j-1),(p+1)*m*m-m)
    else:
        b = numpy.reshape(numpy.eye(1,(p+1)*m*m-m,i*m*m+(k-1)*m+j),(p+1)*m*m-m)

    return numpy.zeros(n), b, 0

# Find core reactions of a delayed kinetic system (Y',M0,...,Mp)
def CoreReactions(P, L, Y, p, m, n, dyn_equiv):
    invT, M, A = DenseRealization(P, L, Y, p, m, n, dyn_equiv)
    Ec = numpy.zeros((0, 3))
    H = [j for j in range(n)] + [j for j in range(n+(p+1)*n*m,n+(p+1)*n*m+(p+1)*m*m-m)]

    for i in range(p+1):
        for j in range(m):
            for k in range(m):
                if i == 0 and j == k:
                    continue
                if A[i][j][k] > 0:
                    Lc = [l for l in L] + [ZeroEdge(p, m, n, i, j, k)]
                    Q, B = FindPositive(P, Lc, Y, p, m, n, H, dyn_equiv)
                    if len([j for j in range(n) if j not in B]) > 0:
                        Ec = numpy.insert(Ec, Ec.shape[0], numpy.array([i, j, k]), axis=0)
    return Ec

# Find all realizations
def AllRealizations(P, L, Y, p, m, n, dyn_equiv):
    Real = DenseRealization(P, L, Y, p, m, n, dyn_equiv)
    invT = Real[0]
    M = Real[1]
    A = Real[2]
    m = M.shape[2]

    Ec = CoreReactions(P, L, Y, p, m, n, dyn_equiv)
    Enc = numpy.zeros((0, 3))
    for i in range(p+1):
        for j in range(m):
            for k in range(m):
                if A[i][j][k] > 0 and FindRow(Ec, numpy.array([i, j, k])) >= Ec.shape[0]:
                    Enc = numpy.insert(Enc, Enc.shape[0], numpy.array([i, j, k]), axis=0)
    Enc = Enc.astype(int)

    # Calculate the number of non-core edges
    N = len(Enc)

    # Put the dense realization to the last index of everything
    D = numpy.ones(N)
    Realizations = {}
    Realizations[str(Dec(D))] = Real

    S = [[] for j in range(N+1)]
    S[N].append(D)

    for k in range(N, -1, -1):
        while(len(S[k]) > 0):
            R = S[k].pop()
            startTime = time.time()
            LR = [l for l in L]
            for r in range(N):
                if R[r] == 0:
                    LR = [l for l in LR] + [ZeroEdge(p, m, n, Enc[r][0], Enc[r][1], Enc[r][2])]
            for r in range(N):
                if R[r] != 0:
                    LRr = [l for l in LR] + [ZeroEdge(p, m, n, Enc[r][0], Enc[r][1], Enc[r][2])]
                    Real = DenseRealization(P, LRr, Y, p, m, n, dyn_equiv)
                    if not numpy.all(numpy.any(Real[0] > 0, axis=0)):
                        continue
                    U = Binarize(Real[2], Enc)
                    if str(Dec(U)) not in Realizations:
                        Realizations[str(Dec(U))] = Real
                        S[int(sum(U))].append(U)
            print(Dec(R), " done in ", time.time()-startTime)

    return M, Realizations, ReactionNumber(A)

# Find all realizations
def AllRealizationsM(Mp, L, Yp, p, m, n, generate, dyn_equiv):
    Y, Real = DenseRealizationM(Mp, L, Yp, p, m, n, generate, dyn_equiv)
    M = numpy.matmul(Real[0], Real[1])
    m = M.shape[2]

    P = M2P(M, Y, p, m, n)

    return Y, AllRealizations(P, L, Y, p, m, n, dyn_equiv)

def DFS(A, v, visited, components):
    visited[v] = True
    components[len(components)-1].append(v)
    for w in range(len(visited)):
        if A[v][w] == 1 and visited[w] == False:
            DFS(A, w, visited, components)

def FillOrder(A, v, visited, stack, components):
    visited[v] = True
    for w in range(len(visited)):
        if A[v][w] == 1 and visited[w] == False:
            FillOrder(A, w, visited, stack, components)
    stack = stack.append(v)

def FindCrossEdges(Ap, p, m):
    A = numpy.copy(Ap)

    for j in range(m):
        A[0][j][j] = 0

    A[0] = numpy.sum(A, axis=0)
    for j in range(m):
        for k in range(m):
            if A[0][j][k] > 0:
                A[0][j][k] = 1
    A = numpy.transpose(A[0])

    stack = []
    visited = [False] * m
    components = [[]]

    for v in range(m):
        if visited[v] == False:
            FillOrder(A, v, visited, stack, components)

    A = numpy.transpose(A)
    visited = [False] * m

    while stack:
        v = stack.pop()
        if visited[v] == False:
            DFS(A, v, visited, components)
            components.append([])
    components.pop()
    A = numpy.transpose(A)

    cross_edges = []
    for c1 in components:
        for c2 in components:
            if c1 == c2:
                continue
            for cc1 in c1:
                for cc2 in c2:
                    if A[cc1][cc2] == 1:
                        cross_edges.append([cc1, cc2])

    return cross_edges

def WRRealization(P, L, Y, p, m, n, dyn_equiv):
    invT, M, A = DenseRealization(P, L, Y, p, m, n, dyn_equiv)

    m = M.shape[2]

    count = 0
    cross_edges = FindCrossEdges(A, p, m)
    while cross_edges != []:
        print("Number of cross edges", len(cross_edges))
        for cross_edge in cross_edges:
            for i in range(p+1):
                L = L + [ZeroEdge(p, m, n, i, cross_edge[1], cross_edge[0])]
        M, invT, A = DenseRealization(P, L, Y, p, m, n, dyn_equiv)
        cross_edges = FindCrossEdges(A, p, m)

    if ReactionNumber(A) == 0:
        print("There is no weakly reversible linearly conjugate realization.")
        return numpy.zeros((n,n)), numpy.zeros((p+1,n,m)), numpy.zeros((p+1,m,m))
    return invT, M, A

def WRRealizationM(Mp, L, Yp, p, m, n, generate, dyn_equiv):
    Y, Real = DenseRealizationM(Mp, L, Yp, p, m, n, generate, dyn_equiv)
    invT = Real[0]
    M = Real[1]
    A = Real[2]

    m = M.shape[2]

    P = M2P(M, Y, p, m, n)

    print("Starting WR")

    return WRRealization(P, L, Y, p, m, n, dyn_equiv)

def AllWRRealizations(P, L, Y, p, m, n, dyn_equiv):
    Real = WRRealization(P, L, Y, p, m, n, dyn_equiv)
    invT = Real[0]
    M = Real[1]
    A = Real[2]
    m = M.shape[2]

    Ec = CoreReactions(P, L, Y, p, m, n, dyn_equiv)
    Enc = numpy.zeros((0, 3))
    for i in range(p+1):
        for j in range(m):
            for k in range(m):
                if A[i][j][k] > 0 and FindRow(Ec, numpy.array([i, j, k])) >= Ec.shape[0]:
                    Enc = numpy.insert(Enc, Enc.shape[0], numpy.array([i, j, k]), axis=0)
    Enc = Enc.astype(int)

    # Calculate the number of non-core edges
    N = len(Enc)

    # Put the dense realization to the last index of everything
    D = numpy.ones(N)
    Realizations = {}
    Realizations[str(Dec(D))] = Real

    S = [[] for j in range(N+1)]
    S[N].append(D)

    for k in range(N, -1, -1):
        while(len(S[k]) > 0):
            R = S[k].pop()
            startTime = time.time()
            LR = [l for l in L]
            for r in range(N):
                if R[r] == 0:
                    LR = [l for l in LR] + [ZeroEdge(p, m, n, Enc[r][0], Enc[r][1], Enc[r][2])]
            for r in range(N):
                if R[r] != 0:
                    LRr = [l for l in LR] + [ZeroEdge(p, m, n, Enc[r][0], Enc[r][1], Enc[r][2])]
                    Real = WRRealization(P, LRr, Y, p, m, n, dyn_equiv)
                    if not numpy.all(numpy.any(Real[0] > 0, axis=0)):
                        continue
                    U = Binarize(Real[2], Enc)
                    if str(Dec(U)) not in Realizations:
                        Realizations[str(Dec(U))] = Real
                        S[int(sum(U))].append(U)
            print(Dec(R), " done in ", time.time()-startTime)

    return M, Realizations, ReactionNumber(A)

# Create a tikz graph
def RealToTikz(Y, invT, A, filename):
    colors = ['blue', 'red', 'green']
    coeff = lambda c, j, p: ('+' if (j > 0 and c > 0 and p) else '') + (str(int(c)) + 'X_' + str(j + 1) if (c != 1 and c != 0) else ('X_' + str(j + 1) if c == 1 else ''))
    bend = lambda i, t, c: (',bend right=' + str(int(10 * (t - (t - 1) % 2) - 20 * c))) if t > 1 else ''
    dash = lambda i: (',dashed,' + colors[i - 1]) if i > 0 else ''
    loop_in = lambda i, m, c: str(180 + 360 / m * i + 180 / m - (m - 2) * 180 / m / 2 - c * 90)
    loop_out = lambda i, m, c: str(180 + 360 / m * i + 180 / m - (m - 2) * 180 / m / 2 - (c + 1) * 90)
#    loop_in = lambda i, m, c: str(360 / m * i + 180 / m + 45)
#    loop_out = lambda i, m, c: str(360 / m * i + 180 / m - 45)

    p = A.shape[0]
    m = A.shape[1]
    n = Y.shape[0]

    tikz = '\\begin{tikzpicture}[scale = 1.2, every node/.style={scale=1.2}]\n'

    for i in range(m):
        tikz += '\\node[complex] (C' + str(i)
#        tikz += ') at (cos(pi/' + str(m) + '*' + str(i) + '),sin(pi/' + str(m) + '*' + str(i) + ')) {$'
        angle = 360 / m * i + 180 / m
        tikz += ') at (' + str(3.9 * numpy.cos(numpy.deg2rad(angle))) + ',' + str(3 * numpy.sin(numpy.deg2rad(angle))) + ') {$'
        plus = False
        cx = coeff(Y[0][i], 0, plus)
        if cx != '':
            plus = True
        for j in range(1, n):
            tmp = coeff(Y[j][i], j, plus)
            if tmp != '':
                cx += tmp
                plus = True
        if cx != '':
            tikz += cx
        else:
            tikz += '0'

        tikz += '$};\n'

    for j in range(m):
        for k in range(m):
            total = len([A[l][j][k] for l in range(p) if A[l][j][k] > 0]) 
            counter = 0
            for i in range(p):
                if A[i][j][k] > 0:
                    if j != k:
                        tikz += '\\path[-latex] + (C' + str(k) + ') edge[' + bend(i, total, counter) + dash(i) + ',very thick] node[] {} (C' + str(j) + ');\n'
                    else:
                        tikz += '\\path[-latex] + (C' + str(k) + ') edge[loop below,min distance=10mm,in=' + loop_in(j, m, counter) + ',out=' + loop_out(j, m, counter) + ',looseness=6' + dash(i) + ',very thick] (C' + str(j) + ');\n'
                    counter += 1
    tikz += '\\end{tikzpicture}'

    with open(filename, 'a+') as f:  
        f.write(tikz)

def MatrixToTex(M, isint):
    coeff = lambda c, i: str(int(c)) if i else (str(c) if c != 0 else '0')

    matrix = '\\begin{bmatrix}\n'
    for i in range(M.shape[0]):
        matrix += '\t'
        for j in range(M.shape[1] - 1):
            matrix += coeff(M[i][j], isint) + ' & '
        matrix += coeff(M[i][-1], isint) + '\\\\\n'
    matrix += '\\end{bmatrix}\\\n'

    return matrix
