# Incompact3D example
In this example we process some data from Incompact3D. How to run the example
1. Install gdown package to get the data: `pip install -f requirements.txt` or `pip install gdown`
2. To download the example `python get_data.py`
3. To run the example `python example.py`

## Overview of `example.py`

1. The coordinate and instantaneous velocity data is extracted using the function `extract_xdmf`. 
2. A `CoordStruct` is created from the coordinate data
3. FlowStructND is created from instantaneous velocity data
4. Spanwise-streamwise averaged velocity is produced using the `reduce` method
5. Velocity fluctuations computed by subtracting the averaged velocity from the instantaneous velocity
6. Contour plots are created for the velocity fluctuations
7. `slice` method applied to `FlowStructND` to reduce domain size and create slices for plotting in pyvista
8. FlowStructND converted to `pyvista.StructuredGrid`
9. Streak isosurfaces plotted and displayed