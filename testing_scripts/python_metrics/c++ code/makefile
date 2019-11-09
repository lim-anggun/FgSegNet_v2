CFLAGS=-Wall -Wextra --pedantic -g0 -O3 -std=c++0x
OPENCV=`pkg-config opencv --cflags --libs`

all: comparator

comparator:
	g++ -o comparator main.cpp VideoFolder.cpp Comparator.cpp $(OPENCV) $(CFLAGS)

clean:
	rm -rf *o comparator method