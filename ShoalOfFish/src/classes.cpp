#include "classes.h"

static unsigned int CompileShader(const std::string& source, unsigned int type)
{
	unsigned int id = glCreateShader(type);
	const char* src = source.c_str();
	check_opengl_err(glShaderSource(id, 1, &src, nullptr));
	check_opengl_err(glCompileShader(id));

	int result;
	check_opengl_err(glGetShaderiv(id, GL_COMPILE_STATUS, &result));
	if (result == GL_FALSE)
	{
		int length;
		check_opengl_err(glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length));
		char* message = (char*)_malloca(length * sizeof(char));
		check_opengl_err(glGetShaderInfoLog(id, length, &length, message));
		std::cout << "Failed to compile " << (type == GL_SHADER_TYPE ? "vertex" : "fragment") << std::endl;
		std::cout << message << std::endl;
		check_opengl_err(glDeleteShader(id));
		return 0;
	}

	return id;
}

static unsigned int CreateShader(const std::string& vertexShader, const std::string& sourceShader)
{
	unsigned int program = glCreateProgram();
	unsigned int vs = CompileShader(vertexShader, GL_VERTEX_SHADER);
	unsigned int fs = CompileShader(sourceShader, GL_FRAGMENT_SHADER);

	check_opengl_err(glAttachShader(program, vs));
	check_opengl_err(glAttachShader(program, fs));
	check_opengl_err(glLinkProgram(program));
	check_opengl_err(glValidateProgram(program));

	check_opengl_err(glDeleteShader(vs));
	check_opengl_err(glDeleteShader(fs));

	return program;
}

// Shader
Shader::Shader()
{
	std::string fragmentCode = "#version 330 core\n"
		"layout (location = 0) out vec4 FragColor; \n"
		"\n"
		"void main()\n"
		" { \n"
		" FragColor = vec4(0.8f, 0.3f, 0.02f, 1.0f); \n"
		"}\n";

	std::string vertexCode = "#version 330 core\n"
		"layout (location = 0) in vec3 aPos;\n"
		"\n"
		"void main()\n"
		"{\n"
		"	gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
		"}\n";

	id = CreateShader(vertexCode, fragmentCode);
}

void Shader::Activate()
{
	check_opengl_err(glUseProgram(id));
}

void Shader::Delete()
{
	check_opengl_err(glDeleteProgram(id));
}

// VBO

VBO::VBO(unsigned int size)
{
	check_opengl_err(glGenBuffers(1, &id));
	check_opengl_err(glBindBuffer(GL_ARRAY_BUFFER, id));
	check_opengl_err(glBufferData(GL_ARRAY_BUFFER, size, nullptr, GL_DYNAMIC_DRAW));
}

void VBO::Bind()
{
	check_opengl_err(glBindBuffer(GL_ARRAY_BUFFER, id));
}

void VBO::Unbind()
{
	check_opengl_err(glBindBuffer(GL_ARRAY_BUFFER, 0));
}

void VBO::Delete()
{
	check_opengl_err(glDeleteBuffers(1, &id));
}

// VAO

VAO::VAO()
{
	check_opengl_err(glGenVertexArrays(1, &id));
}

void VAO::LinkVBO(VBO& vbo, unsigned int layout)
{
	vbo.Bind();
	check_opengl_err(glVertexAttribPointer(layout, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0));
	check_opengl_err(glEnableVertexAttribArray(layout));
	vbo.Unbind();
}

void VAO::Bind()
{
	check_opengl_err(glBindVertexArray(id));
}

void VAO::Unbind()
{
	check_opengl_err(glBindVertexArray(0));
}

void VAO::Delete()
{
	check_opengl_err(glDeleteVertexArrays(1, &id));
}
