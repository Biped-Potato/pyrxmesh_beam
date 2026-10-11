import pyrxmesh as rx
import pyrxmesh_beam
import polyscope as ps
import numpy as np

def main()->None:
    rx.init(0)
    ps.init()
    spacing = 1.0
    num_vertices_x = 6
    num_vertices_y = 5
    vertices = np.empty(shape=(num_vertices_x*num_vertices_y, 3),dtype="float32")

    face_shape = (num_vertices_y - 1, (num_vertices_x - 1) * 2)
    face_count = face_shape[0] * face_shape[1]
    faces = np.empty(shape=(face_count,3),dtype="uint32")

    vertex_index = 0
    for y in range(num_vertices_y):
        for x in range(num_vertices_x):
            vertices[vertex_index] = [x * spacing + ((y+1) % 2) * (spacing * 0.5), (num_vertices_y - y) * spacing,0]
            vertex_index += 1

    face_index = 0
    #input is top of triangle
    def create_upwards_triangle(x,y,even=0):
        top          =  y      * num_vertices_x + x + even
        bottom_left  = (y + 1) * num_vertices_x + x
        bottom_right = (y + 1) * num_vertices_x + x + 1

        faces[face_shape[1] * y + 2 * x] = [top,bottom_left,bottom_right]
    
    #input is top left of triangle
    def create_downwards_triangle(x,y,even=0):
        bottom    = (y + 1) * num_vertices_x + x + even
        top_left  = y * num_vertices_x + x
        top_right = y * num_vertices_x + x + 1

        faces[face_shape[1] * y + 2 * x + 1] = [top_left,bottom,top_right]
    
    for y in range(0, num_vertices_y - 1, 2):    
        for x in range(num_vertices_x - 1):
            create_upwards_triangle(x,y)
            create_downwards_triangle(x,y,1)
        
    for y in range(1, num_vertices_y - 1, 2):    
        for x in range(num_vertices_x - 1):
            create_downwards_triangle(x,y)
            create_upwards_triangle(x,y,1)
    
    print(face_index,face_count)
    ps.register_surface_mesh("mesh", vertices, faces, edge_width=1.0,edge_color=(0.0,0.0,0.0))
    ps.show()

if __name__ == "__main__":
    main()