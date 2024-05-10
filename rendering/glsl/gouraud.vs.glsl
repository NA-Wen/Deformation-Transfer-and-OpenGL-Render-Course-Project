#version 330 core
out vec3 LightingColor;

layout (location = 0) in vec3 aPos;   // 位置变量的属性位置值为 0
layout (location = 1) in vec3 aNormal; // normal变量的属性位置值为 1

uniform vec3 lightPos;  // 光源位置
uniform vec3 viewPos;   // 观察位置
uniform vec3 objectColor;
uniform vec3 lightColor;
uniform mat4 model;     // 模型矩阵
uniform mat4 view;      // 视图矩阵
uniform mat4 projection;// 投影矩阵

void main()
{
    // 环境光
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    // 漫反射
    vec3 norm = mat3(transpose(inverse(model))) * aNormal;
    norm = normalize(norm);
    vec3 FragPos = vec3(model * vec4(aPos, 1.0));
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // 镜面反射
    float specularStrength = 0.2;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * lightColor * spec;

    LightingColor = ambient + diffuse + specular;

    gl_Position = projection * view * model * vec4(aPos, 1.0); 

}

