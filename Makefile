all: use_matrices

#For debugging
OPT=-g -Wall -std=c++11 -O
# need -std::c++11 to use initializer_list
# -O for optimistaion

#All objects (except main) come from cpp and hpp
%.o:	%.cpp %.hpp
	g++ ${OPT} -c -o $@ $<
#use_vectors relies on objects which rely on headers
use_vectors:	use_vectors.cpp Vector.o Exception.o
		g++ ${OPT} -o use_vectors use_vectors.cpp Vector.o Exception.o
use_matrices: use_matrices.cpp Matrix.o Exception.o Vector.o
		g++ ${OPT} -o use_matrices use_matrices.cpp Matrix.o Exception.o Vector.o
clean:
	# rm -f *.o *~ use_vectors
	rm -f *.o *~ use_matrices
