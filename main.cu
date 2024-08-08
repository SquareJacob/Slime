#include <SDL.h>
#include <SDL_image.h>
#include <SDL_ttf.h>
#include <SDL_mixer.h>
#include <iostream>
#include <stdlib.h>  
#include <crtdbg.h>   //for malloc and free
#include <set>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>
#define _CRTDBG_MAP_ALLOC
#ifdef _DEBUG
#define new new( _NORMAL_BLOCK, __FILE__, __LINE__)
#endif

SDL_Window* window;
SDL_Renderer* renderer;
bool running;
SDL_Event event;
std::set<std::string> keys;
std::set<std::string> currentKeys;
int mouseX = 0;
int mouseY = 0;
int mouseDeltaX = 0;
int mouseDeltaY = 0;
int mouseScroll = 0;
std::set<int> buttons;
std::set<int> currentButtons;
const int WIDTH = 800;
const int HEIGHT = 600;

__global__ void initCurand(unsigned int seed, curandState* state) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	curand_init(seed, idx, 0, &state[idx]);
}

const double TRAILDECAY = 0.01;
const double DIFFUSION = 20.0; //inverse
double pheremones[HEIGHT * WIDTH] = { 0.0 };
double newP[HEIGHT * WIDTH];
double *d_newP, *d_pheremones;
size_t s_pheremones = sizeof(double) * static_cast<size_t>(WIDTH) * static_cast<size_t>(HEIGHT);
__global__ void diffuseTrail(double* pheremones, double* newP, double DIFFUSION) {
	int i = blockIdx.x;
	int j = threadIdx.x;
	int x, y;
	double sum = 0.0;
	for (int k = -1; k < 2; k++) {
		for (int l = -1; l < 2; l++) {
			x = i + k;
			y = j + l;
			if (k == 0 && l == 0) {
				sum += pheremones[y * WIDTH + x] * DIFFUSION;
			}
			else if (-1 < x && x < WIDTH && -1 < y && y < HEIGHT) {
				sum += pheremones[y * WIDTH + x];
			}
		}
	}
	newP[j * WIDTH + i] = sum / (8.0 + DIFFUSION);
	//newP[j * WIDTH + i] = 1.0;
}
__global__ void copyTrail(double* pheremones, double* newP) {
	pheremones[threadIdx.x * WIDTH + blockIdx.x] = newP[threadIdx.x * WIDTH + blockIdx.x];
	//pheremones[threadIdx.x * WIDTH + blockIdx.x] = 1.0;
}

class Cell {
public:
	double x = 0.0, y = 0.0, angle = 0.0;
	__device__ bool move(double speed) {
		double deltaX = speed * cos(angle);
		double deltaY = speed * sin(angle);
		if (0.0 < x + deltaX && x + deltaX < WIDTH && 0.0 < y + deltaY && y + deltaY < HEIGHT) {
			x += deltaX;
			y += deltaY;
			return true;
		}
		else {
			return false;
		}
	}
	__device__ void sense(curandState* state, double sensorDistance, double sensorAngle, double rotateAmount, double* pheremones) {
		double frontSensor = pheremones[static_cast<int>(y + sensorDistance * sin(angle) + 0.5) * WIDTH + static_cast<int>(x + sensorDistance * cos(angle) + 0.5)];
		double leftSensor = pheremones[static_cast<int>(y + sensorDistance * sin(angle + sensorAngle) + 0.5) * WIDTH + static_cast<int>(x + sensorDistance * cos(angle + sensorAngle) + 0.5)];
		double rightSensor = pheremones[static_cast<int>(y + sensorDistance * sin(angle - sensorAngle) + 0.5) * WIDTH + static_cast<int>(x + sensorDistance * cos(angle - sensorAngle) + 0.5)];
		if (frontSensor > leftSensor && frontSensor > rightSensor) {
			return;
		}
		else if (frontSensor < leftSensor && frontSensor < rightSensor) {
			angle += static_cast<float>(2 * curand(state) % 2 - 1) * rotateAmount;
		}
		else if (rightSensor > leftSensor) {
			angle -= rotateAmount;
		}
		else if (rightSensor < leftSensor) {
			angle += rotateAmount;
		}
	}
	void draw() {
		SDL_RenderDrawPoint(renderer, static_cast<int>(x), static_cast<int>(y));
	}
	__device__ void trail(double* pheremones) {
		pheremones[static_cast<int>(y) * WIDTH + static_cast<int>(x)] = 1.0;
	}
};
double speed = 1.0;
double sensorDistance = 10.0;
double sensorAngle = M_PI / 4;
double rotateAmount = M_PI / 16;
const int CELLCOUNT = 10000; //KEEP SQUARE AND LESS THAN 1024^2
const int CELLCOUNTSQRT = 100; //KEEP AS SQRT OF CELLCOUNT
Cell cells[CELLCOUNT];
Cell* d_cells;
size_t s_cells = sizeof(Cell) * static_cast<size_t>(CELLCOUNT);

__global__ void moveCell(Cell* cells, curandState* state, double speed, double* pheremones) {
	int i = CELLCOUNTSQRT * threadIdx.x + blockIdx.x;
	if (cells[i].move(speed)) {
		cells[i].trail(pheremones);
	}
	else {
		cells[i].angle = curand_uniform(state) * 2.0 * M_PI;
	}
}
__global__ void sense(Cell* cells, curandState* state, double sensorDistance, double sensorAngle, double rotateAmount, double* pheremones) {
	cells[CELLCOUNTSQRT * threadIdx.x + blockIdx.x].sense(state, sensorDistance, sensorAngle, rotateAmount, pheremones);
}

void debug(int line, std::string file) {
	std::cout << "Line " << line << " in file " << file << ": " << SDL_GetError() << std::endl;
}

double random() {
	return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

__device__ Uint32 red = 0x01000000, blue = 0x00010000, green = 0x00000100;
__global__ void pixelize(double* pheremones, Uint32* pixel_ptr, double TRAILDECAY) {
	double* p = &pheremones[threadIdx.x * WIDTH + blockIdx.x];
	if (*p > 0.0) {
		*p = *p - TRAILDECAY;
		if (*p < 0.0) {
			*p = 0.0;
		}
	}
	pixel_ptr[threadIdx.x * WIDTH + blockIdx.x] = static_cast<Uint32>(*p * 255) * (red + green + blue) + 255;
}
Uint32* pixel_ptr, *d_pixel_ptr, *pixel_ptrA;
size_t s_pixel_ptr = sizeof(Uint32) * static_cast<size_t>(WIDTH * HEIGHT);

Uint32 frameStart, calcStart, drawStart;
int frameTime = 0;
bool timing = true;
int main(int argc, char* argv[]) {
	srand(time(0));
	if (SDL_Init(SDL_INIT_EVERYTHING) == 0 && TTF_Init() == 0 && Mix_OpenAudio(44100, MIX_DEFAULT_FORMAT, 2, 2048) == 0) {
		//Setup
		window = SDL_CreateWindow("Window", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, 0);
		if (window == NULL) {
			debug(__LINE__, __FILE__);
			return 0;
		}

		renderer = SDL_CreateRenderer(window, -1, 0);
		if (renderer == NULL) {
			debug(__LINE__, __FILE__);
			return 0;
		}

		cudaSetDevice(0);
		curandState* d_state;
		cudaMalloc(&d_state, sizeof(curandState));
		initCurand << <1, 1 >> > (time(0), d_state);
		cudaMalloc((void**)&d_pheremones, s_pheremones);
		cudaMalloc((void**)&d_newP, s_pheremones);
		cudaMalloc((void**)&d_cells, s_cells);
		cudaMalloc((void**)&d_pixel_ptr, s_pixel_ptr);

		SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
			SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);
		SDL_Texture* textureA = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
			SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);
		void* txtPixels;
		int pitch;
		SDL_PixelFormat* format = SDL_AllocFormat(SDL_PIXELFORMAT_RGBA8888);

		double angle;
		double radius;
		for (int i = 0; i < CELLCOUNT; i++) {
			angle = random() * 2.0 * M_PI;
			radius = std::min(HEIGHT, WIDTH) * random() / 2;
			cells[i].angle = angle;
			cells[i].x = static_cast<float>(WIDTH) / 2.0 - radius * cos(angle);
			cells[i].y = static_cast<float>(HEIGHT) / 2.0 - radius * sin(angle);
		}

		//Main loop
		running = true;
		while (running) {
			//handle events
			frameStart = SDL_GetTicks();
			for (std::string i : keys) {
				currentKeys.erase(i); //make sure only newly pressed keys are in currentKeys
			}
			for (int i : buttons) {
				currentButtons.erase(i); //make sure only newly pressed buttons are in currentButtons
			}
			mouseScroll = 0;
			while (SDL_PollEvent(&event)) {
				switch (event.type) {
				case SDL_QUIT:
					running = false;
					break;
				case SDL_KEYDOWN:
					if (!keys.contains(std::string(SDL_GetKeyName(event.key.keysym.sym)))) {
						currentKeys.insert(std::string(SDL_GetKeyName(event.key.keysym.sym)));
					}
					keys.insert(std::string(SDL_GetKeyName(event.key.keysym.sym))); //add keydown to keys set
					break;
				case SDL_KEYUP:
					keys.erase(std::string(SDL_GetKeyName(event.key.keysym.sym))); //remove keyup from keys set
					break;
				case SDL_MOUSEMOTION:
					mouseX = event.motion.x;
					mouseY = event.motion.y;
					mouseDeltaX = event.motion.xrel;
					mouseDeltaY = event.motion.yrel;
					break;
				case SDL_MOUSEBUTTONDOWN:
					if (!buttons.contains(event.button.button)) {
						currentButtons.insert(event.button.button);
					}
					buttons.insert(event.button.button);
					break;
				case SDL_MOUSEBUTTONUP:
					buttons.erase(event.button.button);
					break;
				case SDL_MOUSEWHEEL:
					mouseScroll = event.wheel.y;
					break;
				}
			}

			calcStart = SDL_GetTicks();
			cudaMemcpy(d_pheremones, pheremones, s_pheremones, cudaMemcpyHostToDevice);
			cudaMemcpy(d_newP, newP, s_pheremones, cudaMemcpyHostToDevice);
			cudaMemcpy(d_cells, cells, s_cells, cudaMemcpyHostToDevice);
			diffuseTrail << <WIDTH, HEIGHT >> > (d_pheremones, d_newP, DIFFUSION);
			copyTrail << <WIDTH, HEIGHT >> > (d_pheremones, d_newP);
			moveCell << <CELLCOUNTSQRT, CELLCOUNTSQRT >> > (d_cells, d_state, speed, d_pheremones);
			sense << <CELLCOUNTSQRT, CELLCOUNTSQRT >> > (d_cells, d_state, sensorDistance, sensorAngle, rotateAmount, d_pheremones);
			cudaDeviceSynchronize();
			cudaMemcpy(pheremones, d_pheremones, s_pheremones, cudaMemcpyDeviceToHost);
			cudaMemcpy(newP, d_newP, s_pheremones, cudaMemcpyDeviceToHost);
			cudaMemcpy(cells, d_cells, s_cells, cudaMemcpyDeviceToHost);
			if (timing) {
				std::cout << "calc time: " << SDL_GetTicks() - calcStart;
			}

			drawStart = SDL_GetTicks();
			SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
			SDL_RenderClear(renderer);
			SDL_LockTexture(texture, NULL, &txtPixels, &pitch);
			pixel_ptr = (Uint32*)txtPixels;

			cudaMemcpy(d_pixel_ptr, pixel_ptr, s_pixel_ptr, cudaMemcpyHostToDevice);
			pixelize << <WIDTH, HEIGHT >> > (d_pheremones, d_pixel_ptr, TRAILDECAY);
			cudaDeviceSynchronize();
			cudaMemcpy(pixel_ptr, d_pixel_ptr, s_pixel_ptr, cudaMemcpyDeviceToHost);

			SDL_UnlockTexture(texture);
			//SDL_RenderCopy(renderer, texture, NULL, NULL);
			SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
			for (int i = 0; i < CELLCOUNT; i++) {
				cells[i].draw();
			}
			SDL_RenderPresent(renderer);
			if (timing) {
				std::cout << " draw time: " << SDL_GetTicks() - drawStart;
			}
			frameTime = SDL_GetTicks() - frameStart;
			if (timing) {
				std::cout << " total time: " << frameTime << std::endl;
			}
		}

		//Clean up
		SDL_FreeFormat(format);
		SDL_DestroyTexture(texture);
		cudaFree(d_pheremones);
		cudaFree(d_newP);
		if (window) {
			SDL_DestroyWindow(window);
		}
		if (renderer) {
			SDL_DestroyRenderer(renderer);
		}
		TTF_Quit();
		Mix_Quit();
		IMG_Quit();
		SDL_Quit();
		return 0;
	}
	else {
		return 0;
	}
}