#include <chrono>
#include <iostream>
#ifndef TIMER_H_
#define TIMER_H_

#include <ctime>
#include <chrono>
using namespace std::chrono;

using namespace std;

void drawText(const std::string& text, float x, float y, void* font)
{
	// Save state
	glPushAttrib(GL_CURRENT_BIT | GL_ENABLE_BIT);

	glDisable(GL_TEXTURE_2D);
	glDisable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);

	glColor3f(.1f, .1f, .1f); // drop shadow
	// Shift shadow one pixel to the lower right.
	glWindowPos2f(x + 1.0f, y - 1.0f);
	auto disPlayFunc = [&font](char chararector){glutBitmapCharacter(font, chararector); };

	std::for_each(text.begin(), text.end(), disPlayFunc);
	glColor3f(.95f, .95f, .95f);        // main text
	glWindowPos2f(x, y);
	std::for_each(text.begin(), text.end(), disPlayFunc);
	// Restore state

	glEnable(GL_TEXTURE_2D);
	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
	glPopAttrib();
}

struct Timer { // High performance timer using standard c++11 chrono
	double elapsed_time_milliseconds = 0;
	high_resolution_clock::time_point t1;
	high_resolution_clock::time_point t2;
	double fps;
	inline Timer() {
	}

	inline void start() {
		t1 = high_resolution_clock::now();
	}
	inline void stop() {
		t2 = high_resolution_clock::now();
		elapsed_time_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	}
	float getFps()
	{
		static double last_frame_time = 0;
		static int frame_count = 0;
		static char fps_text[32];
		t2 = high_resolution_clock::now();
		elapsed_time_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		++frame_count;
		if (elapsed_time_milliseconds > 0.5) {
			fps = frame_count *1000/ elapsed_time_milliseconds;
			t1 = t2;
			frame_count = 0;
		}
		return fps;
	}

};

#endif