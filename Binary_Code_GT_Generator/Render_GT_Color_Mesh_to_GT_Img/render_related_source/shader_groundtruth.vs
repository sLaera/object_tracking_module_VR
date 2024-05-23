#version 420 core
layout (location = 0) in vec3 aPos; // Posizione del vertice

void main()
{
    gl_Position = vec4(aPos, 1.0); // Trasforma la posizione del vertice nello spazio degli schermi
}
