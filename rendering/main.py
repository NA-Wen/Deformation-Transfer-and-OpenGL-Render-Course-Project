import glfw
from OpenGL.GL import *
import numpy as np
import glm
import shader as shader
import numpy as np
from OpenGL.GL import *
import glfw
import argparse

def generate_lookat_matrix(points, camera_pos=None):
    global scene_center
    scene_center = np.mean(points, axis=0)
    camera_position = scene_center - np.array([0, 0, 400]) if camera_pos is None else camera_pos
    camera_target = scene_center
    camera_up = np.array([0, 1, 0])
    return glm.lookAt(glm.vec3(camera_position), glm.vec3(camera_target), glm.vec3(camera_up))


def average_vertices(origin_vertices):
    sum_x = sum_y = sum_z = 0.0
    for vertex in origin_vertices:
        sum_x += vertex[0]
        sum_y += vertex[1]
        sum_z += vertex[2]
    avg_x = sum_x / len(origin_vertices)
    avg_y = sum_y / len(origin_vertices)
    avg_z = sum_z / len(origin_vertices)
    return [avg_x, avg_y, avg_z]

def min_vertices(origin_vertices):
    min_x = origin_vertices[0][0]
    min_y = origin_vertices[0][1]
    min_z = origin_vertices[0][2]
    for vertex in origin_vertices:
        if vertex[0] < min_x:
            min_x = vertex[0]
        if vertex[1] < min_y:
            min_y = vertex[1]
        if vertex[2] < min_z:
            min_z = vertex[2]
    return [min_x, min_y, min_z]

def get_normal(origin_vertices, faces):
    vertice_normals = np.zeros((len(origin_vertices), 3))
    for face in faces:
        v1 = np.array(origin_vertices[face[0]])
        v2 = np.array(origin_vertices[face[1]])
        v3 = np.array(origin_vertices[face[2]])
        a = v2 - v1
        b = v3 - v1
        tmp = np.cross(a, b)
        vertice_normals[face[0]] += tmp / np.linalg.norm(tmp)
        vertice_normals[face[1]] += tmp / np.linalg.norm(tmp)
        vertice_normals[face[2]] += tmp / np.linalg.norm(tmp)
    normal = vertice_normals/3
    return normal

def load_vertices(filename: str):
    origin_vertices, faces = load_obj(filename)
    triangles = []
    normal = get_normal(origin_vertices, faces)
    for face in faces:
        for indice in face:
            triangles.append(origin_vertices[indice])
            triangles.append(normal[indice])

    vertices = np.array(triangles, dtype=np.float32).flatten()
    old_vertices = np.array([])
    vertices = np.concatenate([vertices, old_vertices])
    return origin_vertices, vertices

def load_obj(filename):
    origin_vertices = []
    x_bias = 0.0
    y_bias = 0.0
    z_bias = 0.0
    vertices = []
    faces = []

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                origin_vertices.append([float(vertex) for vertex in line.strip().split()[1:]])
            elif line.startswith('f '):
                faces.append([int(vertex.split('/')[0]) - 1 for vertex in line.strip().split()[1:]])
    x_bias, y_bias, z_bias = average_vertices(origin_vertices)
    for vertice in origin_vertices:
        vertices.append([vertice[0]-x_bias, vertice[1]-y_bias, vertice[2]-z_bias])
    return vertices, faces

def process_input(window):
    global lightposX, lightposY, lightposZ, shaderProgram
    if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
        glfw.set_window_should_close(window, True)
    elif glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
        lightposZ -= 0.1
    elif glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
        lightposZ += 0.1
    elif glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        lightposY -= 0.1
    elif glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        lightposY += 0.1
    elif glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
        lightposX -= 0.1
    elif glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
        lightposX += 0.1

def key_callback(window, key, scancode, action, mods):
    global shaderProgram
    if key == glfw.KEY_SPACE and action == glfw.PRESS:
        shaderProgram = shaderProgram_phong if shaderProgram!=shaderProgram_phong else shaderProgram_gourand
        print(shaderProgram)

def mouse_button_callback(window, button, action, mods):
    global is_dragging, last_mouse_pos
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            is_dragging = True
            last_mouse_pos = glfw.get_cursor_pos(window)
        elif action == glfw.RELEASE:
            is_dragging = False

def mouse_move_callback(window, xpos, ypos):
    global last_mouse_pos, yaw, pitch, camera_target, camera_up, camera_pos, is_dragging, alpha, beta
    if is_dragging:
        x_offset = xpos - last_mouse_pos[0]
        y_offset = ypos - last_mouse_pos[1]  # reversed since y-coordinates go from bottom to top
        print(x_offset, y_offset, xpos, ypos)
        last_mouse_pos = (xpos, ypos)

        x_offset *= mouse_sensitivity
        y_offset *= mouse_sensitivity
        # print(x_offset*np.pi)
        alpha  += x_offset/1600*np.pi
        beta += y_offset/1200*np.pi
        camera_pos[0] = scale * (scene_center[0] + 400 * np.sin(alpha))
        camera_pos[2] = scale * (scene_center[2] + 400 - 400 * np.cos(alpha) - 400 * np.cos(beta))
        camera_pos[1] = scale * (scene_center[1] + 400 * np.sin(beta))
    
        print(camera_pos)

def scroll_callback(window, xoffset, yoffset):
    global scale, camera_pos
    scale += yoffset*0.1
    camera_pos[0] = scale * (scene_center[0] + 400 * np.sin(alpha))
    camera_pos[2] = scale * (scene_center[2] + 400 - 400 * np.cos(alpha) - 400 * np.cos(beta))
    camera_pos[1] = scale * (scene_center[1] + 400 * np.sin(beta))
    print(scale)




def create_buffer():
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, 8 * vertices.size, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, int(8 * 6), None)
    glEnableVertexArrayAttrib(VAO, 0)
    glVertexAttribPointer(1, 3, GL_DOUBLE, GL_FALSE, int(8 * 6), ctypes.c_void_p(8 * 3))
    glEnableVertexArrayAttrib(VAO, 1)

    return VAO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load vertices from an OBJ file.')
    parser.add_argument('filename', type=str, help='The path to the .obj file')
    args = parser.parse_args()
    
    origin_vertices, vertices = load_vertices(args.filename)
    lightposX, lightposY, lightposZ = 0.0, 0.0, 0.0
    
    glfw.init()
    window = glfw.create_window(1600, 1200, "render", None, None)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL)
    glfw.make_context_current(window)


    VAO = create_buffer()
    
    shaderProgram_phong = shader.Shader("./glsl/phong.vs.glsl", "./glsl/phong.fs.glsl")
    shaderProgram_gourand = shader.Shader("./glsl/gouraud.vs.glsl", "./glsl/gouraud.fs.glsl")
    shaderProgram = shaderProgram_gourand

    glEnable(GL_DEPTH_TEST) 

    last_mouse_pos = (0, 0)
    is_dragging = False
    scene_center = np.mean(origin_vertices, axis=0)

    camera_pos = scene_center - np.array([0, 0, 400])
    camera_target = scene_center
    camera_up  = np.array([0, 1, 0])
    alpha = 0.0
    beta = 0.0
    scale = 1.0
    camera_speed = 0.01
    mouse_sensitivity = 0.5
    
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, mouse_move_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_key_callback(window, key_callback)


    while not glfw.window_should_close(window):
        process_input(window)
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        model = glm.mat4(1.0)
        radius = 10.0
        view  = generate_lookat_matrix(origin_vertices,camera_pos)
        projection = glm.perspective(glm.radians(45.0), 800 / 600, 0.1, 1000.0)

        shaderProgram.use()
        # 获取uniform变量的位置
        lightPosLoc = glGetUniformLocation(shaderProgram.shaderProgram, "lightPos")
        viewPosLoc = glGetUniformLocation(shaderProgram.shaderProgram, "viewPos")
        lightColorLoc = glGetUniformLocation(shaderProgram.shaderProgram, "lightColor")
        objectColorLoc = glGetUniformLocation(shaderProgram.shaderProgram, "objectColor")

        # 设置uniform变量的值
        glUniform3f(lightPosLoc, lightposX, lightposY, lightposZ)
        glUniform3f(viewPosLoc, 0.0, 0.0, 3.0)
        glUniform3f(lightColorLoc, 1.0, 0.8, 0.8)
        glUniform3f(objectColorLoc, 1.0, 1.0, 1.0)

        
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram.shaderProgram, 'model'), 1, GL_FALSE, glm.value_ptr(model))
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram.shaderProgram, 'view'), 1, GL_FALSE, glm.value_ptr(view))
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram.shaderProgram, 'projection'), 1, GL_FALSE, glm.value_ptr(projection))

        glBindVertexArray(VAO)  
        glDrawArrays(GL_TRIANGLES, 0, len(vertices))

        glfw.swap_buffers(window)
        glfw.poll_events()

    shaderProgram.delete()

