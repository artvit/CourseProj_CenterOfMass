from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog


voxel_edge = 0.1
eps = 1e-8
filename = 'input.txt'


def rho(point):
    return point[0] + point[1] * point[2]


def gui():
    global voxel_edge
    global filename

    def openfile():
        global filename
        filename = filedialog.askopenfilename()
        v.set(filename)

    def start():
        global voxel_edge
        voxel_edge = float(e.get())
        root.quit()

    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", exit)

    tk.Button(root, text='Choose file', command=openfile).grid(row=1, column=1)
    v = tk.StringVar()
    v.set(filename)
    tk.Label(root, textvariable=v).grid(row=1, column=2)
    tk.Label(root, text='Step:').grid(row=2, column=1)
    e = tk.Entry(root)
    e.insert(0, str(voxel_edge))
    e.grid(row=2, column=2)
    tk.Button(root, text=' Run ', command=start).grid(row=3, column=1)
    root.mainloop()


gui()


def read_data(file_name):
    vertices = []
    triangles = []
    with open(file_name, 'r') as f:
        for line in f:
            if line == '\n':
                break
            str_arr = line.split(' ')
            vertices.append([float(str_arr[0]), float(str_arr[1]), float(str_arr[2])])
        for line in f:
            str_arr = line.split(' ')
            triangles.append([int(str_arr[0]), int(str_arr[1]), int(str_arr[2])])
    return np.array(vertices, dtype=float), np.array(triangles, dtype=int)


vertices, triangles = read_data(filename)

x = np.array([vertex[0] for vertex in vertices], dtype=float)
y = np.array([vertex[1] for vertex in vertices], dtype=float)
z = np.array([vertex[2] for vertex in vertices], dtype=float)


def edges_list():
    edges = []
    triangles_for_edge = {}
    for triangle in triangles:
        if (triangle[0], triangle[1]) in edges:
            i = edges.index((triangle[0], triangle[1]))
            triangles_for_edge[i].append(triangle)
        elif (triangle[1], triangle[0]) in edges:
            i = edges.index((triangle[1], triangle[0]))
            triangles_for_edge[i].append(triangle)
        else:
            edges.append((triangle[0], triangle[1]))
            i = len(edges) - 1
            triangles_for_edge[i] = [triangle]

        if (triangle[1], triangle[2]) in edges:
            i = edges.index((triangle[1], triangle[2]))
            triangles_for_edge[i].append(triangle)
        elif (triangle[2], triangle[1]) in edges:
            i = edges.index((triangle[2], triangle[1]))
            triangles_for_edge[i].append(triangle)
        else:
            edges.append((triangle[1], triangle[2]))
            i = len(edges) - 1
            triangles_for_edge[i] = [triangle]

        if (triangle[0], triangle[2]) in edges:
            i = edges.index((triangle[0], triangle[2]))
            triangles_for_edge[i].append(triangle)
        elif (triangle[2], triangle[0]) in edges:
            i = edges.index((triangle[2], triangle[0]))
            triangles_for_edge[i].append(triangle)
        else:
            edges.append((triangle[0], triangle[2]))
            i = len(edges) - 1
            triangles_for_edge[i] = [triangle]

    return [[edge, triangles_for_edge[edges.index(edge)][0], triangles_for_edge[edges.index(edge)][1]] for edge in edges]


edges_triangles = edges_list()
edges = [et[0] for et in edges_triangles]


def get_intersection_with_surface(pnt, tr):
    p1 = vertices[tr[0]]
    p2 = vertices[tr[1]]
    p3 = vertices[tr[2]]

    a = p1[1] * (p2[2] - p3[2]) + p2[1] * (p3[2] - p1[2]) + p3[1] * (p1[2] - p2[2])
    if a == 0:
        return None
    b = p1[2] * (p2[0] - p3[0]) + p2[2] * (p3[0] - p1[0]) + p3[2] * (p1[0] - p2[0])
    c = p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])
    d = -(p1[0] * (p2[1] * p3[2] - p3[1] * p2[2]) + p2[0] * (p3[1] * p1[2] - p1[1] * p3[2]) + p3[0] * (
    p1[1] * p2[2] - p2[1] * p1[2]))

    x_inter = (-d - b * pnt[1] - c * pnt[2]) / a

    if x_inter < pnt[0]:
        return None
    return np.array((x_inter, pnt[1], pnt[2]), dtype=float)


def check_in_triangle(pnt, tr):
    def triangle_square(a, b, c):
        p = (a + b + c) / 2
        t = p * (p - a) * (p - b) * (p - c)
        if t < 0:
            return .0
        return math.sqrt(t)

    a, b, c = vertices[tr[0]], vertices[tr[1]], vertices[tr[2]]

    ab = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
    bc = math.sqrt((b[0] - c[0]) ** 2 + (b[1] - c[1]) ** 2 + (b[2] - c[2]) ** 2)
    ca = math.sqrt((a[0] - c[0]) ** 2 + (a[1] - c[1]) ** 2 + (a[2] - c[2]) ** 2)

    ap = math.sqrt((pnt[0] - a[0]) ** 2 + (pnt[1] - a[1]) ** 2 + (pnt[2] - a[2]) ** 2)
    bp = math.sqrt((pnt[0] - b[0]) ** 2 + (pnt[1] - b[1]) ** 2 + (pnt[2] - b[2]) ** 2)
    cp = math.sqrt((pnt[0] - c[0]) ** 2 + (pnt[1] - c[1]) ** 2 + (pnt[2] - c[2]) ** 2)

    s0 = triangle_square(ab, bc, ca)
    s1 = triangle_square(ap, bp, ab)
    s2 = triangle_square(ap, cp, ca)
    s3 = triangle_square(bp, cp, bc)

    diff = s1 + s2 + s3 - s0
    if abs(diff) < eps:
        if s1 == 0:
            return True, (tr[0], tr[1])
        if s2 == 0:
            return True, (tr[0], tr[2])
        if s3 == 0:
            return True, (tr[1], tr[2])
        return True
    else:
        return False


def drawing(voxels, colors, center):
    fig = plt.figure()
    fig.suptitle('Interpolation')
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    ax.plot_trisurf(x, y, z, triangles=triangles, alpha=0.05, color='white')

    xs = np.array([voxel[0] for voxel in voxels], dtype=float)
    ys = np.array([voxel[1] for voxel in voxels], dtype=float)
    zs = np.array([voxel[2] for voxel in voxels], dtype=float)
    ax.scatter(xs, ys, zs, c=colors)

    fig = plt.figure()
    fig.suptitle('Center of mass')
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    ax.plot_trisurf(x, y, z, triangles=triangles, alpha=0.25)
    ax.scatter([center[0]], [center[1]], [center[2]], c='green')

    plt.show()


def density_colors(p):
    min_d = min(p)
    max_d = max(p)
    k = max_d - min_d
    if k < eps:
        return [1, 0, 0]
    colors = []
    for d in p:
        r = (d - min_d) / k
        g = 1.0 - r
        colors.append(np.array((r, g, 0.0), dtype=float))
    return colors


def check(point, edge, x_min):
    t1 = edges_triangles[edges.index(edge)][1]
    t2 = edges_triangles[edges.index(edge)][2]

    q1 = np.array((x_min, point[1] + voxel_edge / 100, point[2]), dtype=float)
    q3 = np.array((x_min, point[1] - voxel_edge / 100, point[2]), dtype=float)

    q2 = np.array((x_min, point[1], point[2] + voxel_edge / 100), dtype=float)
    q4 = np.array((x_min, point[1], point[2] - voxel_edge / 100), dtype=float)

    def check_pair(pnt1, pnt2, tr1, tr2):
        p1 = get_intersection_with_surface(pnt1, tr1)
        p3 = get_intersection_with_surface(pnt2, tr2)
        if p1 is not None and p3 is not None:
            r1 = check_in_triangle(p1, t1)
            r3 = check_in_triangle(p3, t2)
            if r1 is True and r3 is True:
                return True
        return False

    if check_pair(q1, q3, t1, t2):
        return True
    if check_pair(q3, q1, t1, t2):
        return True
    if check_pair(q2, q4, t1, t2):
        return True
    if check_pair(q4, q2, t1, t2):
        return True

    return False


def main():
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    z_min, z_max = min(z), max(z)

    n_x = int((x_max - x_min) / voxel_edge)
    n_y = int((y_max - y_min) / voxel_edge)
    n_z = int((z_max - z_min) / voxel_edge)

    voxels = []

    for j in range(n_y):
        for k in range(n_z):
            for i in range(n_x):
                q = np.array((
                    x_min + i * voxel_edge + voxel_edge / 2,
                    y_min + j * voxel_edge + voxel_edge / 2,
                    z_min + k * voxel_edge + voxel_edge / 2
                ), dtype=float)
                intersection_count = 0
                edge_intersect = 0
                for triangle in triangles:
                    p = get_intersection_with_surface(q, triangle)
                    if p is not None:
                        res = check_in_triangle(p, triangle)
                        if res is True:
                            intersection_count += 1
                            continue
                        elif res is False:
                            continue
                        else:
                            if check(p, res[1], x_min):
                                edge_intersect += 1
                if (intersection_count - edge_intersect // 2) % 2 == 1:
                    voxels.append(q)

    center = np.zeros_like(vertices[0])
    mass = 0
    densities = []
    for voxel in voxels:
        p = rho(voxel)
        densities.append(p)
        mass += p
        center += p * voxel
    center /= mass

    print(center)

    colors = density_colors(densities)
    drawing(voxels, colors, center)


if __name__ == '__main__':
    main()
