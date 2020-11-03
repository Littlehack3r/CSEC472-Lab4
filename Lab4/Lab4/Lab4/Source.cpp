#include "SFML/Graphics/Image.hpp"
#include <string>
#include <iostream>
#include <filesystem>

void searchDir(std::vector<std::string>* images, std::string dir) {
	for (const auto& entry : std::filesystem::directory_iterator(dir)) {
		std::string path = entry.path().filename().string();
		if (path.at(0) == 'f') {
			if (path.substr(path.size() - 4) == ".png")
				images->push_back(entry.path().string());
		}
	}
}

std::vector<std::string> loadPathData() {
	std::vector<std::string> images;

	searchDir(&images, "C:/Users/Isaiah/Desktop/Lab 4/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0");
	searchDir(&images, "C:/Users/Isaiah/Desktop/Lab 4/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_1");
	searchDir(&images, "C:/Users/Isaiah/Desktop/Lab 4/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_2");
	searchDir(&images, "C:/Users/Isaiah/Desktop/Lab 4/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_3");
	searchDir(&images, "C:/Users/Isaiah/Desktop/Lab 4/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_4");
	searchDir(&images, "C:/Users/Isaiah/Desktop/Lab 4/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_5");
	searchDir(&images, "C:/Users/Isaiah/Desktop/Lab 4/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_6");
	searchDir(&images, "C:/Users/Isaiah/Desktop/Lab 4/NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_7");

	return images;
}

std::vector<std::string> splitVector(std::vector<std::string> images, unsigned int start, unsigned int end) {
	std::vector<std::string> split;

	for (unsigned int i = start; i < end; i++)
		split.push_back(images.at(i));

	return split;
}

int** loadMatrix(std::string path) {
	int** matrix = new int* [512];
	for (int i = 0; i < 512; i++)
		matrix[i] = new int[512];

	sf::Image image;
	image.loadFromFile(path);

	for (int x = 0; x < 512; x++) {
		for (int y = 0; y < 512; y++)
			matrix[x][y] = image.getPixel(x, y).toInteger();
	}

	return matrix;
}

void freeMatrix(int** matrix) {
	for (int i = 0; i < 512; i++)
		delete[] matrix[i];
	delete[] matrix;
}

bool method1(float threshold, int** matrix, std::string path) {
	int matching = 0;

	int** matrix1 = matrix;
	int** matrix2 = loadMatrix(path);

	for (int x = 0; x < 512; x++) {
		for (int y = 0; y < 512; y++) {
			if (matrix1[x][y] == matrix2[x][y])
				matching++;
		}
	}

	freeMatrix(matrix2);

	float match = static_cast<float>(matching / (512 * 512));
	bool equal = match >= threshold;
	return equal;
}

bool method2(float threshold, int** matrix, std::string path) {
	int matching = 0;

	int** matrix1 = matrix;
	int** matrix2 = loadMatrix(path);

	for (int i = 0; i < 6; i++) {
		int x_start = rand() % 502;
		int y_start = rand() % 502;
		for (int x = x_start; x < x_start + 10; x++) {
			for (int y = y_start; y < y_start + 10; y++) {
				if (matrix1[x][y] == matrix2[x][y])
					matching++;
			}
		}
	}

	freeMatrix(matrix2);

	float match = static_cast<float>(matching / (10 * 10 * 6));
	bool equal = match >= threshold;
	return equal;
}

bool method3(float threshold, int** matrix, std::string path) {
	int matching = 0;

	int** matrix1 = matrix;
	int** matrix2 = loadMatrix(path);

	for (int i = 0; i < 6; i++) {
		int block[5] = { };
		int x[5] = { };
		x[0] = rand() % 497 + 5;
		x[1] = x[0] - 5;
		x[2] = x[0];
		x[3] = x[0] + 5;
		x[4] = x[0];

		int y[5] = { };
		y[0] = rand() % 497 + 5;
		y[1] = y[0];
		y[2] = y[0] - 5;
		y[3] = y[0];
		y[4] = y[0] + 5;
		for (int j = 0; j < 5; j++) {
			for (int a = x[j]; a < x[j] + 5; a++) {
				for (int b = y[j]; b < y[j] + 5; b++) {
					if (matrix1[a][b] == matrix2[a][b])
						block[j]++;
				}
			}
		}

		int b = block[0];
		for (int k = 1; k < 5; k++) {
			if (block[k] > b)
				b = block[k];
		}

		matching += b;
	}

	freeMatrix(matrix2);

	float match = static_cast<float>(matching / (5 * 5 * 6));
	bool equal = match >= threshold;
	return equal;
}

void compare(std::vector<std::string> trainImages, std::vector<std::string> testImages, bool (*method)(float, int**, std::string)) {
	int perfect = 0;
	int fraud = 0;
	int insult = 0;
	int counter = 0;

	for (auto train : trainImages) {
		int matches = 0;
		testImages.push_back(train);
		int** matrix = loadMatrix(train);
		
		for (auto test : testImages)
			matches += method(0.9f, matrix, test);

		if (matches > 1)
			fraud++;
		else if (matches < 1)
			insult++;
		else
			perfect++;

		testImages.pop_back();
		freeMatrix(matrix);
		counter++;
		std::cout << counter << "/1500" << std::endl;
	}

	std::cout << "Perfect: " << perfect << " | Fraud: " << fraud << " | Insult: " << insult << std::endl;
}

int main(int argc, int** argv) {
	std::vector<std::string> images = loadPathData();
	std::vector<std::string> trainImages = splitVector(images, 0, 1500);
	std::vector<std::string> testImages = splitVector(images, 1500, 2000);
	images.clear();
	std::cout << "Method 1:" << std::endl;
	compare(trainImages, testImages, &method1);
	std::cout << "Method 2:" << std::endl;
	compare(trainImages, testImages, &method2);
	std::cout << "Method 3:" << std::endl;
	compare(trainImages, testImages, &method3);
	return 0;
}