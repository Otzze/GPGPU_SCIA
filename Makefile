SRC=$(wildcard src/*.cu)

build:
	mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make all -j

debug:
	mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Debug .. && make all -j

clean:
	$(RM) -r *.bin a.out $(OBJ) build

claen: clean # j'en peux plus
