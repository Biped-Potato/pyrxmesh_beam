import src.beam as beam

import pyrxmesh as rx
import torch
import pyrxmesh_beam
import polyscope as ps
import polyscope.imgui as psim
import numpy as np


def main()->None:
    rx.init(0)
    torch.cuda.set_device(0)
    ps.init()

    beam_width = 50
    beam_height = 10
    beam_spacing = 1.0

    anim_offset = 10.0

    mesh = rx.RXMeshStatic(*beam.generate_beam(beam_width, beam_height, beam_spacing))

    rest_shape = mesh.add_face_attribute("fRestShape", dtype="float32", dim=4)
    coordinates = mesh.input_vertex_coordinates()
    line_search_coordinates = mesh.add_vertex_attribute("ls", dtype="float32", dim=3)
    
    pyrxmesh_beam.compute_rest_shape(mesh,coordinates,rest_shape)

    coords = coordinates.to_torch("device").detach().requires_grad_(True)

    #offset right side of beam
    offset_linear = torch.zeros_like(coords)

    offset_linear[beam_width - 1::beam_width,1] = anim_offset
    offset_global = offset_linear[mesh.linear_to_global(rx.ElementKind.Vertex)]

    with torch.no_grad():
        coords.add_(offset_global)

    energy = pyrxmesh_beam.make_energy(mesh,rest_shape)

    lower, upper = mesh.bounding_box()

    left_bound = float(lower[0]) + torch.finfo(coords.dtype).tiny
    right_bound = float(upper[0]) - torch.finfo(coords.dtype).tiny
    
    free_mask =  (coords[:,0] > left_bound) & (coords[:,0] < right_bound)
    
    gradient = torch.empty_like(coords,memory_format=torch.contiguous_format)

    ps.register_surface_mesh("initial_mesh", coords.detach().cpu().numpy(), mesh.faces(), edge_width=1.0,edge_color=(0.0,0.0,0.0),enabled=False)
    
    steps = 10000
    #lr = 1e-2
    step = 0
    max_tries = 100
    line_search = line_search_coordinates.to_torch("device")
    m = ps.register_surface_mesh("mesh", coords.detach().cpu().numpy(), mesh.faces(), edge_width=1.0,edge_color=(0.0,0.0,0.0))
    m.add_scalar_quantity("boundary_condition",free_mask.detach().cpu().numpy(),'vertices')

    
    def animate():
        nonlocal step
        if step >= steps:
            return
        loss = energy.value_and_grad(coords,out=gradient,copy="never",gradient_mask=free_mask)

        lr = 1.0
        tries_left = max_tries
        tolerance = 1e-7
        while tries_left > 0:
            with torch.no_grad():
                torch.add(coords, gradient, alpha=-lr,out=line_search)
                ls_loss = energy.evaluate(line_search_coordinates)
                if ls_loss < float(loss) and abs(loss - ls_loss) > tolerance:
                    coords.add_(gradient, alpha=-lr)
                   # print(ls_loss, float(loss))
                    break
                lr = lr / 2
                tries_left -= 1
        else:
            print("exit")
            step = steps
            return
        
        rx.cuda_stream_synchronize()
        step += 1
        m.update_vertex_positions(coords.detach().cpu().numpy())
    ps.set_user_callback(animate)
    ps.show()

if __name__ == "__main__":
    main()