import flowpy2 as fp2
import flowpy2.plotting as cplt
import xml.etree.ElementTree as ET
import numpy as np
import os
import pyvista as pv

from typing import List

def extract_xdmf(path_root: str,comps: List[str], snapshot: int):
        
    # read xml file to get meta data
    fn = os.path.join(path_root, 'snapshot-%d.xdmf'%snapshot)
    root =ET.parse(fn).getroot()

    shape_str = root.find('Domain/Topology').get('Dimensions')
    shape = tuple(int(s) for s in shape_str.split())[::-1]

    geom = root.find('Domain/Geometry').findall('DataItem')

    geom_data = [None]*3

    data = geom[0].text.replace('\n','').split()
    geom_data[0] = np.array([np.float64(x) for x in data])

    data = geom[1].text.replace('\n','').split()
    geom_data[1] = np.array([np.float64(x) for x in data])

    data = geom[2].text.replace('\n','').split()
    geom_data[2] = np.array([np.float64(x) for x in data])


    files = ['%s-%d.bin'%(comp,snapshot) for comp in comps]
    data  = np.zeros((len(comps), *shape))
    for i, file in enumerate(files):
            data[i] = np.fromfile(os.path.join(path_root, file),
                                dtype='f8').reshape(shape,
                                                        order='F')

    return geom_data, data

def main():
    # Run get_data.py first
    path = 'incompact_channel/data'

    comps = ['ux', 'uy', 'uz']
    geom_data, inst_data = extract_xdmf(path, comps, 10)

    coords = fp2.CoordStruct('Cartesian',
                             geom_data,
                             index=['x','y','z'])
    
    inst_fs = fp2.FlowStructND(coords,
                          inst_data,
                          comps=['u', 'v', 'w'])
    
    print(inst_fs)

    # average in the x and z directions
    fs_avg  = inst_fs.reduce(np.mean, ('x','z'))
    print(fs_avg)

    # Create fluctuation fstruct by removing average
    fluct_fs = inst_fs.copy()
    for comp in fluct_fs.comps:
          fluct_fs[comp] -= fs_avg[comp][:,None]

    #plot fluctuation contour
    fig, ax = cplt.subplots(3)

    # pcolormesh
    qm_list = [None]*3
    for i, comp in enumerate(fluct_fs.comps):
        qm_list[i] = fluct_fs.pcolormesh(comp,'xz', 0.05, ax=ax[i],
                                 cmap='seismic')
        
        # create colorbar
        cax = fig.colorbar(qm_list[i], ax=ax[i])
        cax.set_label(r"$%s'$"%comp)

    # Set colorbar limits 
    qm_list[0].set_clim(-0.2,0.2)
    qm_list[1].set_clim(-0.08,0.08)
    qm_list[2].set_clim(-0.1,0.1)

    # set axis labels
    for a in ax:
        a.set_ylabel(r"$z$")

    ax[-1].set_xlabel(r"$x$")

    fig.tight_layout()
    cplt.show()

    # example using pyvista
    p = pv.Plotter()

    # Remove top half of domain (keep y=0-1)
    fluct_fs_slice = fluct_fs.slice[:,:1,:]

    # Create planes in x and z near the boundary
    ## This could also be done in pyvista using the slice filter
    fluct_fs_plane_x = fluct_fs.slice[7.9,:,:]
    fluct_fs_plane_z = fluct_fs.slice[:,:,0.05]

    # Convert to VTK (pyvista.StructuredGrid)
    fluct_vtk = fluct_fs_slice.to_vtk()
    
    # create isosurfaces
    streaks = fluct_vtk.contour([-0.1,0.1], scalars='u')

    # add to plotter
    p.add_mesh(streaks,
               cmap=['b','g'],
               show_scalar_bar=False)
    
    p.add_mesh(fluct_fs_plane_x.to_vtk(),
               scalars='u',
               cmap='seismic',
               clim=(-0.25,0.25),
               show_scalar_bar=False)
    
    p.add_mesh(fluct_fs_plane_z.to_vtk(),
               scalars='u',
               cmap='seismic',
               clim=(-0.25,0.25),
               show_scalar_bar=False)
    
    # set camera
    cpos = list(p.camera.focal_point)
    cpos[0] -= 15
    p.camera.position = cpos
    p.camera.up = (0,1,0)
    p.camera.azimuth = 30
    p.camera.elevation = 20

    p.show()

if __name__ == '__main__':
     main()