import numpy as np
from numpy.typing import NDArray

def generate_beam(num_vertices_x, num_vertices_y, spacing) -> tuple[NDArray[np.float32],NDArray[np.uint32]]:
    vertices = np.empty(shape=(num_vertices_x*num_vertices_y, 3),dtype="float32")

    face_shape = (num_vertices_y - 1, (num_vertices_x - 1) * 2)
    face_count = face_shape[0] * face_shape[1]
    faces = np.empty(shape=(face_count,3),dtype="uint32")

    for y in range(num_vertices_y):
        for x in range(num_vertices_x):
            vertices[y * num_vertices_x + x] = [x  * spacing, (num_vertices_y - y) * spacing,0]
    
    face_index = 0
    #input is top left of triangle
    def create_bottom_triangle(x,y):
        top_left     =  y      * num_vertices_x + x
        bottom_left  = (y + 1) * num_vertices_x + x
        bottom_right = (y + 1) * num_vertices_x + x + 1

        faces[face_shape[1] * y + 2 * x + 1] = [top_left,bottom_left,bottom_right]
    #input is top left of triangle
    def create_top_triangle(x,y):
        bottom_right = (y + 1) * num_vertices_x + x + 1
        top_left     =  y      * num_vertices_x + x
        top_right    =  y      * num_vertices_x + x + 1

        faces[face_shape[1] * y + 2 * x] = [top_left,bottom_right,top_right]

    #face order is by row and top triangle is before bottom triangle
    for y in range(0, num_vertices_y - 1):    
        for x in range(num_vertices_x - 1):
            create_top_triangle(x,y)
            create_bottom_triangle(x,y)
    return (vertices,faces)
