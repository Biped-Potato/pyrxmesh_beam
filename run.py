import src.beam as beam

import pyrxmesh as rx
import torch
import pyrxmesh_beam
import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import matplotlib.pyplot as plt

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
    
    pyrxmesh_beam.compute_rest_shape(mesh,coordinates,rest_shape)

    coords = coordinates.to_torch("device").detach().requires_grad_(True)

    #offset right side of beam
    offset_linear = torch.zeros_like(coords)

    offset_linear[beam_width - 1::beam_width,1] = anim_offset
    offset_global = offset_linear[mesh.linear_to_global(rx.ElementKind.Vertex)]

    with torch.no_grad():
        coords.add_(offset_global)

    initial_coords = coords.detach().clone()

    energy = pyrxmesh_beam.make_energy(mesh,rest_shape)

    lower, upper = mesh.bounding_box()

    left_bound = float(lower[0]) + torch.finfo(coords.dtype).tiny
    right_bound = float(upper[0]) - torch.finfo(coords.dtype).tiny
    
    free_mask =  (coords[:,0] > left_bound) & (coords[:,0] < right_bound)
    
    gradient = torch.empty_like(coords,memory_format=torch.contiguous_format)

    ps.register_surface_mesh("initial_mesh", coords.detach().cpu().numpy(), mesh.faces(), edge_width=1.0,edge_color=(0.0,0.0,0.0),enabled=False)
    
    #returns time and final energy
    def simulate(steps) -> tuple[float, float]:
        with torch.no_grad():
            coords.copy_(initial_coords)
        max_tries = 100

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for i in range(steps):
            loss = energy.value_and_grad(coords,out=gradient,copy="never",gradient_mask=free_mask)
            
            lr = 1.0
            tries_left = max_tries
            tolerance = 1e-7
            while tries_left > 0:
                with torch.no_grad():
                    coords.add_(gradient, alpha=-lr)
                    ls_loss = energy.evaluate(coordinates)
                    if ls_loss >= loss.item() or abs(loss - ls_loss) < tolerance:
                        coords.add_(-gradient, alpha=-lr)
                    else:
                        break
                    lr = lr / 2
                    tries_left -= 1
            else:
                print("exit")
                break
            
            rx.cuda_stream_synchronize()

        end_event.record()
        end_event.synchronize()

        total_time = start_event.elapsed_time(end_event)
        ending_energy = loss.item()
        return (total_time, ending_energy)

    steps_begin = 1000
    steps_end = 25000
    steps_testing_increment = 1000

    times = []
    energies = []
    iterations = []
    for i in range(steps_begin, steps_end+1, steps_testing_increment):
        time, final_energy = simulate(i)
        times.append(time)
        energies.append(final_energy)
        iterations.append(i)
        print(i, "steps complete", "time",time,"final energy",final_energy)

    plt.figure(1)

    plt.plot(iterations, energies, label = "final energy")
    plt.plot(iterations, times, label = "time")
    plt.legend()
    plt.xlabel("Iterations")

    plt.figure(2)
        
    plt.plot(iterations, energies, label = "final energy")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Final Energies")
    
    plt.figure(3)

    plt.plot(iterations, times, label = "time")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Time")


    plt.show()
    
    m = ps.register_surface_mesh("mesh", coords.detach().cpu().numpy(), mesh.faces(), edge_width=1.0,edge_color=(0.0,0.0,0.0))
    m.add_scalar_quantity("boundary_condition",free_mask.detach().cpu().numpy(),'vertices')

    ps.show()

if __name__ == "__main__":
    main()