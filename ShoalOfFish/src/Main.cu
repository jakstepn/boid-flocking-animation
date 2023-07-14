#include "classes.h"
#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include <direct.h>
#include <cuda.h>
#include <math.h>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

// Cuda error check
#define check_cuda_err(ans)                                                                              \
        if (ans != cudaSuccess)																		\
        {																							\
            fprintf(stderr, "Assert: %s %s %d\n", cudaGetErrorString(ans), __FILE__, __LINE__);		\
            exit(ans);																			\
        }	

#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 600
#define VERT_PER_TRIANGLE 3
#define VAL_PER_VERTEX 3

// Global variables

float speed = 0.5f;
float alignmentWeigth = 0.1f;
float cohesionWeigth = 0.1f;
float separationWeigth = 0.11f;
float aquariumWeight = 0.03f;
float viewDistance = 0.1f;
bool animationStopped = false;
bool isTerminated = false;
float wallLen = 1.8f;

// Aquarium

float aquariumVertices[] = {
	// Top Wall
	-wallLen / 2.0f, wallLen / 2.0f, 0.0f,
	wallLen / 2.0f, wallLen / 2.0f, 0.0f,

	// Right Wall
	wallLen / 2.0f, wallLen / 2.0f, 0.0f,
	wallLen / 2.0f, -wallLen / 2.0f, 0.0f,

	// Bottom Wall
	wallLen / 2.0f, -wallLen / 2.0f, 0.0f,
	-wallLen / 2.0f, -wallLen / 2.0f, 0.0f,

	// Left Wall
	-wallLen / 2.0f, -wallLen / 2.0f, 0.0f,
	-wallLen / 2.0f, wallLen / 2.0f, 0.0f,
};

// Vector operations

__host__ __device__ float DistanceFrom(float3 v1, float3 v2)
{
	return sqrt((v1.x - v2.x) * (v1.x - v2.x) + (v1.y - v2.y) * (v1.y - v2.y) + (v1.z - v2.z) * (v1.z - v2.z));
}

__host__ __device__ float3 MultiplyByVal(float3 v1, float val)
{
	v1.x = v1.x * val;
	v1.y = v1.y * val;
	v1.z = v1.z * val;
	return v1;
}

__host__ __device__ float3 AddVec(float3 v1, float3 v2)
{
	v1.x = v1.x + v2.x;
	v1.y = v1.y + v2.y;
	v1.z = v1.z + v2.z;
	return v1;
}

__host__ __device__ float3 DevideByValue(float3 v1, float val)
{
	v1.x = v1.x / val;
	v1.y = v1.y / val;
	v1.z = v1.z / val;
	return v1;
}

__host__ __device__ float3 SubtractVec(float3 v1, float3 v2)
{
	v1.x = v1.x - v2.x;
	v1.y = v1.y - v2.y;
	v1.z = v1.z - v2.z;
	return v1;
}

__host__ __device__ float3 Normalize(float3 v1)
{
	float3 f = float3();
	f.x = f.y = f.z = 0.0f;

	float len = DistanceFrom(v1, f);
	return DevideByValue(v1, len != 0 ? len : 1);
}

// Set random triangle starting position
float* SetTrianglePositions(int numberOfTriangles, float triangleWidth, float triangleHeight)
{
	// In [0, numberOfTriangles) are located X position values, later from [numberOfTriangles, 2*numberOfTriangles)
	// are Y values.
	float* positions = new float[numberOfTriangles * 2];

	srand(time((time_t*)0));

	for (int i = 0; i < numberOfTriangles; i++)
	{
		float x = ((rand() % 100) / 100.0f * (wallLen - 2*triangleWidth)) - (wallLen - 2*triangleWidth) / 2.0f;
		float y = ((rand() % 100) / 100.0f * (wallLen - 2*triangleHeight)) - (wallLen - 2*triangleHeight) / 2.0f;

		positions[i] = x;
		positions[numberOfTriangles + i] = y;
	}

	return positions;
}

float* CreateVelocities(int numberOfTriangles)
{
	srand(time((time_t*)0));

	// Set random starting velocities
	float* velocities = new float[numberOfTriangles * 2];

	for (int i = 0; i < numberOfTriangles; i++)
	{
		float x = (rand() % 100) / 100.0f - 0.5f;
		float y = (rand() % 100) / 100.0f - 0.5f;

		velocities[i] = x;
		velocities[numberOfTriangles + i] = y;
	}

	return velocities;
}

// Indexes
__host__ __device__ unsigned int GetXIndex(int triangleId) {
	return triangleId;
}
__host__ __device__ unsigned int GetYIndex(int triangleId, int numberOfTriangles) {
	return numberOfTriangles + triangleId;
}

// Values
__host__ __device__ float3 GetVelocity(int triangleId, int numberOfTriangles, float* velocities) {
	float3 velocity = float3();
	velocity.x = velocities[GetXIndex(triangleId)];
	velocity.y = velocities[GetYIndex(triangleId, numberOfTriangles)];
	velocity.z = 0.0f;

	return velocity;
}
__host__ __device__ float3 GetPos(int triangleId, int numberOfTriangles, float* position) {
	float3 pos = float3();
	pos.x = position[GetXIndex(triangleId)];
	pos.y = position[GetYIndex(triangleId, numberOfTriangles)];
	pos.z = 0.0f;

	return pos;
}

// Flocking logic
// Based on the article: 
// https://gamedevelopment.tutsplus.com/tutorials/3-simple-rules-of-flocking-behaviors-alignment-cohesion-and-separation--gamedev-3444

__host__ __device__ float3 AlignmentVec(int triangleId, float* positions,
	int numberOfTriangles, float* velocities, float viewRadius)
{
	float3 point = float3();
	point.x = point.y = point.z = 0;

	float neighborCount = 0.0f;

	// Triangle position
	float3 myTrPos = GetPos(triangleId, numberOfTriangles, positions);

	for (int i = 0; i < numberOfTriangles; i++)
	{
		float3 currTrPos = GetPos(i, numberOfTriangles, positions);

		if (i != triangleId && DistanceFrom(myTrPos, currTrPos) < viewRadius)
		{
			float3 currTrVel = GetVelocity(i, numberOfTriangles, velocities);

			point = AddVec(point, currTrVel);
			neighborCount++;
		}
	}
	if (neighborCount == 0) return point;

	point = DevideByValue(point, neighborCount);
	return Normalize(point);
}

__host__ __device__ float3 CohesionVec(int triangleId, float* positions,
	int numberOfTriangles, float* velocities, float viewRadius)
{
	float3 point = float3();
	point.x = point.y = point.z = 0;

	float neighborCount = 0.0f;

	// Triangle position
	float3 myTrPos = GetPos(triangleId, numberOfTriangles, positions);

	for (int i = 0; i < numberOfTriangles; i++)
	{
		float3 currTrPos = GetPos(i, numberOfTriangles, positions);

		if (i != triangleId && DistanceFrom(myTrPos, currTrPos) < viewRadius)
		{
			point = AddVec(point, currTrPos);
			neighborCount++;
		}
	}
	if (neighborCount == 0) return point;

	point = DevideByValue(point, neighborCount);
	point = SubtractVec(point, myTrPos);
	return Normalize(point);
}

__host__ __device__ float3 SeparationVec(int triangleId, float* positions,
	int numberOfTriangles, float* velocities, float viewRadius)
{
	float3 point = float3();
	point.x = point.y = point.z = 0;

	float neighborCount = 0.0f;

	// Triangle position
	float3 myTrPos = GetPos(triangleId, numberOfTriangles, positions);

	for (int i = 0; i < numberOfTriangles; i++)
	{
		float3 currTrPos = GetPos(i, numberOfTriangles, positions);

		if (i != triangleId && DistanceFrom(myTrPos, currTrPos) < viewRadius)
		{
			point = AddVec(point, SubtractVec(currTrPos, myTrPos));
			neighborCount++;
		}
	}
	if (neighborCount == 0) return point;

	point = DevideByValue(point, neighborCount);
	point = MultiplyByVal(point, -1);
	return Normalize(point);
}

__host__ __device__ float3 SteerFromAquarium(int triangleId, float* positions,
	int numberOfTriangles, float* velocities, float viewRadius,
	float aquariumWitdh, float aquariumHeight)
{
	float halvedAqWidth = aquariumWitdh / 2.0f;
	float halvedAqHeight = aquariumHeight / 2.0f;

	// Triangle position
	float3 myTrPos = GetPos(triangleId, numberOfTriangles, positions);

	// Distance from a wall
	float distX = halvedAqWidth - abs(myTrPos.x);
	float distY = halvedAqHeight - abs(myTrPos.y);

	float3 vecX = float3();
	vecX.y = vecX.x = vecX.z = 0.0f;
	if (viewRadius >= distX)
	{
		vecX.y = vecX.z = 0;
		vecX.x = distX / halvedAqWidth;
	}

	float3 vecY = float3();
	vecY.y = vecY.x = vecY.z = 0.0f;
	if (viewRadius >= distY)
	{
		vecY.x = vecY.z = 0;
		vecY.y = distY / halvedAqHeight;
	}

	float3 point = AddVec(MultiplyByVal(vecY, -(myTrPos.y / abs(myTrPos.y))), MultiplyByVal(vecX, -(myTrPos.x / abs(myTrPos.x))));

	return Normalize(point);
}

__host__ __device__ float3 GetFlockingVelocity(int triangleId, float* positions, float speed,
	int numberOfTriangles, float* velocities, float viewRadius, float alignmentWeight, float cohesionWeight, float separationWeight,
	float steerAquariumWeight, float aquariumWitdh, float aquariumHeight)
{
	float3 alignment = AlignmentVec(triangleId, positions, numberOfTriangles,
		velocities, viewRadius);

	float3 cohesion = CohesionVec(triangleId, positions, numberOfTriangles,
		velocities, viewRadius);

	float3 separation = SeparationVec(triangleId, positions, numberOfTriangles,
		velocities, viewRadius);

	float3 aquarium = SteerFromAquarium(triangleId, positions, numberOfTriangles,
		velocities, viewRadius, aquariumWitdh, aquariumHeight);

	float3 myTrVelocity = GetVelocity(triangleId, numberOfTriangles, velocities);

	float3 res = AddVec(MultiplyByVal(separation, separationWeight), MultiplyByVal(aquarium, steerAquariumWeight));
	res = AddVec(MultiplyByVal(cohesion, cohesionWeight), res);
	res = AddVec(MultiplyByVal(alignment, alignmentWeight), res);
	res = AddVec(myTrVelocity, res);

	return MultiplyByVal(Normalize(res), speed);
}

// Check if new position exceeds the window
__host__ __device__ bool HasReachedWindowBorder(float3 pos, float triangleWidth, float triangleHeight, float aquariumWidth, float aquariumHeight)
{
	if (pos.x - triangleWidth / 2.0f < -aquariumWidth / 2.0f || pos.x + triangleWidth / 2.0f > aquariumWidth / 2.0f ||
		pos.y - triangleHeight / 2.0f < -aquariumHeight / 2.0f || pos.y + triangleHeight / 2.0f > aquariumHeight / 2.0f)
		return true;

	return false;
}

__host__ __device__ float GetAngleFromModel(float3 velocity)
{
	// Triangle direction vector in model space (0, 1, 0)
	// Dot product / (Length v1 * Length v2) = cosx
	float angle = atan2(-velocity.x, velocity.y);
	return angle;
}

__host__ __device__ float3 Rotate(float angle, float3 v)
{
	float cs = cos(angle), sn = sin(angle);
	float tmpx = v.x * cs - v.y * sn;
	float tmpy = v.x * sn + v.y * cs;

	v.x = tmpx;
	v.y = tmpy;

	return v;
}

__host__ __device__ float3* Move(float aquariumWidth, float aquariumHeight, float3 velocity, float3 pos, float triangleWidth, float triangleHeight)
{
	float3 finalVelocity = velocity;
	float3 finalPos = AddVec(pos, finalVelocity);

	// Triangle in model space

	// First vertice
	float3 v0 = float3();
	v0.x = -triangleWidth / 2.0f;
	v0.y = -triangleHeight / 2.0f;
	v0.z = 0.0f;

	// Second vertice
	float3 v1 = float3();
	v1.x = triangleWidth / 2.0f;
	v1.y = -triangleHeight / 2.0f;
	v1.z = 0.0f;

	// Third vertice
	float3 v2 = float3();
	v2.x = 0.0f;
	v2.y = triangleHeight / 2.0f;
	v2.z = 0.0f;

	if (HasReachedWindowBorder(finalPos, triangleWidth, triangleHeight, aquariumWidth, aquariumHeight))
	{
		float3 middleVector = float3();
		middleVector.x = pos.x;
		middleVector.y = pos.y;
		middleVector.z = 0.0f;

		// Move towards middle
		float3 newVelocity = MultiplyByVal(Normalize(MultiplyByVal(middleVector, -1.0f)), 0.001f);

		finalPos = AddVec(pos, newVelocity);

		finalVelocity = newVelocity;
	}

	float3 translation = float3();
	translation.x = pos.x;
	translation.y = pos.y;
	translation.z = 0.0f;

	// Add vector to translate a vertice
	v0 = AddVec(Rotate(GetAngleFromModel(finalVelocity), v0), translation);

	v1 = AddVec(Rotate(GetAngleFromModel(finalVelocity), v1), translation);

	v2 = AddVec(Rotate(GetAngleFromModel(finalVelocity), v2), translation);

	return new float3[5]{ finalPos, v0, v1, v2, finalVelocity };
}

__host__ __device__ void UpdatePositions(unsigned int triangleId, float* positions, float* velocities, int numberOfTriangles, float speed,
	float alignmentWeight, float cohesionWeight, float separationWeight, float steerAquariumWeight, float viewRadius, float windowWidth, float windowHeight, float deltaTime,
	float triangleWidth, float triangleHeight, float* result)
{
	float3 flockingVelocity = MultiplyByVal(GetFlockingVelocity(triangleId, positions, speed, numberOfTriangles, velocities,
		viewRadius, alignmentWeight, cohesionWeight, separationWeight, steerAquariumWeight, windowWidth, windowHeight), deltaTime);

	// Triangle position
	float3 pos = GetPos(triangleId, numberOfTriangles, positions);

	float3* newPositions = Move(windowWidth, windowHeight, flockingVelocity, pos, triangleWidth, triangleHeight);

	// Velocity
	velocities[triangleId] = newPositions[4].x;
	velocities[triangleId + numberOfTriangles] = newPositions[4].y;

	// Save new positions

	// Triangle
	positions[triangleId] = newPositions[0].x;
	positions[triangleId + numberOfTriangles] = newPositions[0].y;

	// Vertices

	// v0
	result[triangleId * (VERT_PER_TRIANGLE * VAL_PER_VERTEX) + 0] = newPositions[1].x;
	result[triangleId * (VERT_PER_TRIANGLE * VAL_PER_VERTEX) + 1] = newPositions[1].y;
	result[triangleId * (VERT_PER_TRIANGLE * VAL_PER_VERTEX) + 2] = newPositions[1].z;

	// v1
	result[triangleId * (VERT_PER_TRIANGLE * VAL_PER_VERTEX) + 3] = newPositions[2].x;
	result[triangleId * (VERT_PER_TRIANGLE * VAL_PER_VERTEX) + 4] = newPositions[2].y;
	result[triangleId * (VERT_PER_TRIANGLE * VAL_PER_VERTEX) + 5] = newPositions[2].z;

	// v2
	result[triangleId * (VERT_PER_TRIANGLE * VAL_PER_VERTEX) + 6] = newPositions[3].x;
	result[triangleId * (VERT_PER_TRIANGLE * VAL_PER_VERTEX) + 7] = newPositions[3].y;
	result[triangleId * (VERT_PER_TRIANGLE * VAL_PER_VERTEX) + 8] = newPositions[3].z;

	delete[] newPositions;
}

// One thread per triangle
__global__ void CalculateVelocity(float* positions, float* velocities, int numberOfTriangles, float speed,
	float alignmentWeight, float cohesionWeight, float separationWeight, float steerAquariumWeight, float viewRadius, float windowWidth, float windowHeight, float deltaTime,
	float triangleWidth, float triangleHeight, float* result)
{
	unsigned int gid = threadIdx.x + blockIdx.x * blockDim.x;

	if (gid < numberOfTriangles)
	{
		UpdatePositions(gid, positions, velocities, numberOfTriangles, speed, alignmentWeight, cohesionWeight,
			separationWeight, steerAquariumWeight, viewRadius, windowWidth, windowHeight, deltaTime, triangleWidth, triangleHeight, result);
	}
}

void CalculateVelocityHost(int triangleId, float* positions, float* velocities, int numberOfTriangles, float speed,
	float alignmentWeight, float cohesionWeight, float separationWeight, float steerAquariumWeight, float viewRadius, float windowWidth, float windowHeight, float deltaTime,
	float triangleWidth, float triangleHeight, float* result)
{
	UpdatePositions(triangleId, positions, velocities, numberOfTriangles, speed, alignmentWeight, cohesionWeight,
		separationWeight, steerAquariumWeight, viewRadius, windowWidth, windowHeight, deltaTime, triangleWidth, triangleHeight, result);
}

// Change viewport when the window is resized
void window_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

// Key press actions
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	// Close application
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
	{
		glfwSetWindowShouldClose(window, GLFW_TRUE);
		isTerminated = true;
	}

	if (key == GLFW_KEY_Q && action == GLFW_PRESS)
		alignmentWeigth += 0.01f;

	if (key == GLFW_KEY_W && action == GLFW_PRESS)
		cohesionWeigth += 0.01f;

	if (key == GLFW_KEY_E && action == GLFW_PRESS)
		separationWeigth += 0.01f;

	if (key == GLFW_KEY_A && action == GLFW_PRESS)
		alignmentWeigth -= 0.01f;

	if (key == GLFW_KEY_S && action == GLFW_PRESS)
		cohesionWeigth -= 0.01f;

	if (key == GLFW_KEY_D && action == GLFW_PRESS)
		separationWeigth -= 0.01f;

	if (key == GLFW_KEY_R && action == GLFW_PRESS)
		aquariumWeight += 0.01f;

	if (key == GLFW_KEY_F && action == GLFW_PRESS)
		aquariumWeight -= 0.01f;

	// Stop animation
	if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
		animationStopped = !animationStopped;
}

// Command line options usage
void usage()
{
	std::cout << "Usage: ShoalOfFish [numberOfFish=10240] [-c]" << std::endl;
	std::exit(EXIT_FAILURE);
}

int main(int argc, char* argv[])
{
	if (argc > 3)
	{
		usage();
	}

	int numOfTriangles = 10240;
	bool useCPU = false;

	if (argc >= 2)
	{
		if (std::string(argv[1]).compare("-c") != 0)
		{
			numOfTriangles = std::atoi(argv[1]);
		}
		else if (argc == 3)
		{
			usage();
		}

		if (argc == 3 && std::string(argv[2]).compare("-c") == 0)
		{
			useCPU = true;
		}
		else if (argc == 3)
		{
			usage();
		}
	}

	float triangleWidth = 0.01f;
	float triangleHeight = 0.03f;

	float* d_positions;
	float* d_velocities;
	float* d_result;

	int num_threads = 1024, num_blocks;
	num_blocks = (numOfTriangles + num_threads - 1) / num_threads;

	int framesCount = 0;
	double lastFPSMeasureTime = 0;

	const int width = WINDOW_WIDTH;
	const int height = WINDOW_HEIGHT;

	// Timer
	cudaEvent_t start, stop;
	float time;
	check_cuda_err(cudaEventCreate(&start));
	check_cuda_err(cudaEventCreate(&stop));

	// Data creation time
	check_cuda_err(cudaEventRecord(start, 0));

	float* positions = SetTrianglePositions(numOfTriangles, triangleWidth, triangleHeight);
	float* velocities = CreateVelocities(numOfTriangles);

	float* result = new float[numOfTriangles * VERT_PER_TRIANGLE * VAL_PER_VERTEX] {0};

	check_cuda_err(cudaEventRecord(stop, 0));
	check_cuda_err(cudaEventSynchronize(stop));
	check_cuda_err(cudaEventElapsedTime(&time, start, stop));

	std::cout << "Data creation time: " << time << "s" << std::endl;

	/* Initialize the library */
	if (!glfwInit())
		return -1;

	/* Create a windowed mode window and its OpenGL context */
	GLFWwindow* window = glfwCreateWindow(width, height, "ShoalOfFish", NULL, NULL);
	glfwSetKeyCallback(window, key_callback);

	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	/* Make the window's context current */
	glfwMakeContextCurrent(window);

	gladLoadGL();

	glViewport(0, 0, width, height);

	Shader shaderProgram = Shader();

	// VAO Vertex array object, VBO Vertex buffer object
	VAO vao1;
	vao1.Bind();

	VBO vbo1(sizeof(aquariumVertices) + sizeof(float) * numOfTriangles * VERT_PER_TRIANGLE * VAL_PER_VERTEX);

	vao1.LinkVBO(vbo1, 0);
	vao1.Unbind();

	glfwSetWindowSizeCallback(window, window_size_callback);

	if (!useCPU)
	{
		// Data Copy
		check_cuda_err(cudaEventRecord(start, 0));
		check_cuda_err(cudaMalloc(&d_positions, sizeof(float) * numOfTriangles * 2));
		check_cuda_err(cudaMalloc(&d_velocities, sizeof(float) * numOfTriangles * 2));
		check_cuda_err(cudaMalloc(&d_result, sizeof(float) * numOfTriangles * VERT_PER_TRIANGLE * VAL_PER_VERTEX));

		check_cuda_err(cudaMemcpy(d_positions, positions, sizeof(float) * numOfTriangles * 2, cudaMemcpyHostToDevice));
		check_cuda_err(cudaMemcpy(d_velocities, velocities, sizeof(float) * numOfTriangles * 2, cudaMemcpyHostToDevice));
		check_cuda_err(cudaMemcpy(d_result, result, sizeof(float) * numOfTriangles * VERT_PER_TRIANGLE * VAL_PER_VERTEX, cudaMemcpyHostToDevice));
		check_cuda_err(cudaEventRecord(stop, 0));

		check_cuda_err(cudaEventSynchronize(stop));
		check_cuda_err(cudaEventElapsedTime(&time, start, stop));

		std::cout << "Data copying time: " << time << "s" << std::endl;
	}

	check_cuda_err(cudaEventDestroy(start));
	check_cuda_err(cudaEventDestroy(stop));

	double lastFrameTime = 0;

	while (!glfwWindowShouldClose(window) && !isTerminated)
	{
		check_opengl_err(glClearColor(0.01f, 0.13f, 0.17f, 1.0f));
		check_opengl_err(glClear(GL_COLOR_BUFFER_BIT));

		double currentTime = glfwGetTime();
		float deltaTime = (float)currentTime - (float)lastFrameTime;
		lastFrameTime = currentTime;

		if (!animationStopped)
		{
			if (!useCPU)
			{
				// GPU
				CalculateVelocity <<<num_blocks, num_threads >> > (d_positions, d_velocities, numOfTriangles,
					speed, alignmentWeigth, cohesionWeigth, separationWeigth, aquariumWeight, viewDistance,
					wallLen, wallLen, deltaTime, triangleWidth, triangleHeight, d_result);

				check_cuda_err(cudaDeviceSynchronize());

				check_cuda_err(cudaMemcpy(result, d_result, sizeof(float) * numOfTriangles * VERT_PER_TRIANGLE * VAL_PER_VERTEX, cudaMemcpyDeviceToHost));
			}
			else
			{
				// CPU
				for (int i = 0; i < numOfTriangles; i++)
				{
					CalculateVelocityHost(i, positions, velocities, numOfTriangles,
						speed, alignmentWeigth, cohesionWeigth, separationWeigth, aquariumWeight,
						viewDistance, wallLen, wallLen, deltaTime, triangleWidth, triangleHeight, result);
				}
			}
		}

		// Add vertices to the opengl buffer
		vbo1.Bind();
		check_opengl_err(glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(aquariumVertices), aquariumVertices));
		check_opengl_err(glBufferSubData(GL_ARRAY_BUFFER, sizeof(aquariumVertices), sizeof(float) * numOfTriangles * VERT_PER_TRIANGLE * VAL_PER_VERTEX, result));
		vbo1.Unbind();

		shaderProgram.Activate();

		// Draw Aquarium and triangles
		vao1.Bind();
		check_opengl_err(glDrawArrays(GL_LINES, 0, 8));
		check_opengl_err(glDrawArrays(GL_TRIANGLES, 8, numOfTriangles * VERT_PER_TRIANGLE));

		/* Swap front and back buffers */
		glfwSwapBuffers(window);

		/* Poll for and process events */
		glfwPollEvents();

		framesCount++;

		// Set window title
		if (currentTime - lastFPSMeasureTime >= 1.0) {
			lastFPSMeasureTime += 1.0;
			char title[80];
			int count = std::snprintf(title, 80, "FPS: %d, Align: %.2f, Cohes: %.2f, Separ: %.2f, Aquar: %.2f, CPU: %s",
				framesCount, alignmentWeigth, cohesionWeigth, separationWeigth, aquariumWeight, useCPU ? "TRUE" : "FALSE");
			glfwSetWindowTitle(window, title);
			framesCount = 0;
		}
	}

	// Free memory
	vao1.Delete();
	vbo1.Delete();
	shaderProgram.Delete();

	glfwDestroyWindow(window);

	delete[] positions;
	delete[] velocities;
	delete[] result;

	if (!useCPU)
	{
		check_cuda_err(cudaFree(d_positions));
		check_cuda_err(cudaFree(d_velocities));
		check_cuda_err(cudaFree(d_result));
	}

	glfwTerminate();

	return 0;
}