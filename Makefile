FLAGS := -std=c++20 -Wall -Wextra -Werror -pedantic

RELEASE_CXX := g++
RELEASE_CXXFLAGS := $(FLAGS) -fopenmp -DNDEBUG -O3

TEST_CXX := g++
TEST_CXXFLAGS := $(FLAGS) -fopenmp -O3

DEBUG_CXX := clang++
DEBUG_CXXFLAGS := $(FLAGS) -fsanitize=address -fsanitize=undefined -O0 -g

CXX := $(RELEASE_CXX)
CXXFLAGS := $(RELEASE_CXXFLAGS)

# CXX := $(TEST_CXX)
# CXXFLAGS := $(TEST_CXXFLAGS)

# CXX := $(DEBUG_CXX)
# CXXFLAGS := $(DEBUG_CXXFLAGS)

test: test.o io.o ops.o unet.o Makefile
	$(CXX) $(CXXFLAGS) -o test test.o io.o ops.o unet.o

unet: main.o io.o ops.o unet.o Makefile
	$(CXX) $(CXXFLAGS) -o unet main.o io.o ops.o unet.o

test.o: test.cpp Makefile
	$(CXX) $(CXXFLAGS) -c test.cpp

main.o: main.cpp Makefile
	$(CXX) $(CXXFLAGS) -c main.cpp

io.o: io.h io.cpp Makefile
	$(CXX) $(CXXFLAGS) -c io.cpp

ops.o: ops.h ops.cpp Makefile
	$(CXX) $(CXXFLAGS) -c ops.cpp

unet.o: unet.h unet.cpp Makefile
	$(CXX) $(CXXFLAGS) -c unet.cpp

clean:
	rm -f test test.o main.o io.o ops.o unet.o
