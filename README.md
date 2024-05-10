# Deformation-Transfer-and-OpenGL-Render-Course-Project

> This project is conducted as a course assignment for "Exploration and Practice in Software Engineering (2)", supervised by [Prof. Feng Xu](http://xufeng.site/).

The project comprises two main parts:
- **Deformation Transfer (DF)**: Replication of deformation transfer algorithms. This part of the project can be found under the folder named `df`.
- **OpenGL Rendering**: Implementation of OpenGL rendering for Phong and Gouraud models. This part is located under the folder named `render`.

For running the two projects, setting up environment is necessary. Run the following command to install the necessary dependencies:

```bash
pip install -r requirements.txt
```

---
## Deformation Transfer Project
This project focuses on the deformation transfer of triangular meshes, replicating [Deformation Transfer for Triangle Meshe](https://people.csail.mit.edu/jovan/assets/papers/sumner-2004-dtt.pdf). The method does not require the source and target meshes to have the same number of points or faces. It achieves corresponding deformation transfer by establishing a correspondence between the source and target meshes.

### Running the Code
By default, this implementation will transfer the transformation from `mesh/s_0.obj` to `mesh/s_1.obj` to `mesh/t_0.obj`, and the resulting file will be named `result.obj` under the current working directory.
To execute the code and generate the transformation result `result.obj` under the current directory, follow these steps:

1. Open your terminal.
2. Navigate to the directory containing the project files.
3. Run the following command:

```bash
python transformation.py
```

### Implementation Details
The code structure consists of the main file `transformation.py` and a supporting `meshlib` folder for mesh operations. Below are some important points to note:

1. This is not a complete replication of Deformation Transfer, as the correspondence part is not implemented due to it not being required for the class.
2. For the analysis of the algorithm's robustness to transformations like rotation, some code is commented out to apply transformations on the `s0`, `s1`, and `t0` meshes. The default transformation used in the code is rotation.
3. To address the sensitivity to transformation, the Kabsch algorithm is employed for rotation alignment, while simple first vertex position alignment is used for translation transformation.

All experimental results related to this report are stored in the `mesh` folder for reproducibility.

### Acknowledgements
Instead of using the `OpenMesh` module, the project's implementation of mesh utilities draws significant inspiration from the resources available in [Deformation-Transfer-for-Triangle-Meshes](https://github.com/mickare/Deformation-Transfer-for-Triangle-Meshes), which provides a more lightweight and easily adaptable approach.

---

## OpenGL Rendering Project
This project is an assignment to create a simple rendering tool using OpenGL to visualize the 3D results of Deformation Transfer.
This project implements 3D model loading with Vertex Array Objects (VAO), Vertex Buffer Objects (VBO), and Element Buffer Objects (EBO) and 3D projection, which involves camera models and two shading frequencies based on the Blinn-Phong reflection model: Gouraud and Phong. 
For better observe the phenomenon of rendering, two more features are added:
- Mouse drag for perspective control.
- Keyboard control for switching rendering modes.

### Usage Instructions

#### Running Instructions
To render an object file:
1. Place the `.obj` file you wish to render in the working directory.
2. Execute the following command to render the corresponding `.obj` file:
```bash
python main.py {filename}
```

#### Functionality Description
- **Shading Method Switching**: The default shading is Gouraud. Press the **spacebar** to switch to Phong shading, and press again to switch back to Gouraud.
- **Zooming**: Use the **mouse** wheel to zoom in and out of the object display.
- **Rotation**: Hold the **left mouse button** to rotate the object in the display.
- **Light Position Adjustment**: Use the keyboard's arrow keys and W, S keys to adjust the light source position. **The up and down keys adjust the Z-coordinate of the light, W/S adjust the Y-coordinate, and the left/right keys adjust the X-coordinate.**

### Acknowledgements
The project's implementation of Pythonic style is greatly assisted by the resources found in [基于python的OpenGL](https://www.cnblogs.com/jiujiubashiyi/p/16479817.html).

