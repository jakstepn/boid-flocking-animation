#ifndef SHADER_CLASS_H
#define SHADER_CLASS_H

#include "glad/glad.h"
#include <string>
#include <iostream>

// OpenGL error check
#define check_opengl_err(ans)																		\
	GlClearError();\
	ans;\
	if(!GlCheckError())\
		exit(0);\

// Remove all errors
static void GlClearError()
{
	while (glGetError() != GL_NO_ERROR);
}

static bool GlCheckError()
{
	GLenum err;
	while ((err = glGetError()) != GL_NO_ERROR)
	{
		fprintf(stderr, "OpenGL Assert: %d\n", err);
		return false;
	}
	return true;
}

class Shader
{
public:
	unsigned int id;
	Shader();

	void Activate();
	void Delete();
};
#endif // !SHADER_CLASS_H

#ifndef VERTEX_BUFFER_O
#define VERTEX_BUFFER_O

class VBO
{
public:
	unsigned int id;
	VBO(unsigned int size);

	void Bind();
	void Unbind();
	void Delete();
};

#endif // !VERTEX_BUFFER_O

#ifndef VERTEX_ARR_O
#define VERTEX_ARR_O

class VAO
{
public:
	unsigned int id;
	VAO();

	void LinkVBO(VBO& vbo, unsigned int layout);
	void Bind();
	void Unbind();
	void Delete();
};

#endif // !VERTEX_BUFFER_O
