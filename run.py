import src.beam as beam

import pyrxmesh as rx
import torch
import pyrxmesh_beam
import polyscope as ps
import numpy as np


def main()->None:
    rx.init(0)
    torch.cuda.set_device(0)
    ps.init()

    beam_width = 10
    beam_height = 6
    beam_spacing = 1.0

    anim_offset = 2.0

    mesh = rx.RXMeshStatic(*beam.generate_beam(beam_width, beam_height, beam_spacing))

    rest_shape = mesh.add_face_attribute("fRestShape", dtype="float32", dim=4)
    coordinates = mesh.input_vertex_coordinates()

    pyrxmesh_beam.compute_rest_shape(mesh,coordinates,rest_shape)

    coords = coordinates.to_torch("device").detach().requires_grad_(True)

    #offset right side of beam
    offset = torch.zeros_like(coords)
    offset[beam_width - 1::beam_width,1] = anim_offset

    with torch.no_grad():
        coords.add_(offset)

    energy = pyrxmesh_beam.make_energy(mesh,rest_shape)

    steps = 1000
    gradient = torch.empty(coords.shape, dtype=coords.dtype, device=coords.device)
    #gradient = torch.empty_like(coords,dtype=coords.dtype,device=coords.device)

    lr = 1e-3
    for i in range(steps):
        loss = energy.value_and_grad(coords,out=gradient,copy="never")
        print(loss.detach().cpu().numpy())
        with torch.no_grad():
            coords.add_(gradient, alpha=-lr)
        
    ps.register_surface_mesh("mesh", coords.detach().cpu().numpy(), mesh.faces(), edge_width=1.0,edge_color=(0.0,0.0,0.0))
    
    ps.show()

if __name__ == "__main__":
    main()