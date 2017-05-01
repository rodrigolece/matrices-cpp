#include <iostream>
#include <cassert>
#include "Exception.hpp"
#include "Vector.hpp"
#include "Matrix.hpp"


int main(int argc, char const *argv[]) {

  Matrix my_first_matrix(2,3);
  my_first_matrix(1,1) = 1;
  my_first_matrix(2,2) = 3;
  // std::cout << -my_first_matrix;

  Matrix another_matrix(3,3);
  another_matrix(1,2) = 3;
  another_matrix(1,3) = 4;
  another_matrix(3,2) = -0.5;

  // Matrix result_sum(3,3);
  // result_sum = my_first_matrix - another_matrix;
  // std::cout << result_sum;
  // std::cout <<  my_first_matrix / 3.0;

  // int* s = size(my_first_matrix);
  // std::cout << s[0] << s[1];

  Vector a_vector(3);
  a_vector(1) = 1; a_vector(3) = 3;
  // std::cout << my_first_matrix * a_vector << "\n";
  // std::cout << eye(3) * a_vector << "\n";
  // std::cout << another_matrix * a_vector << "\n";

  std::cout << eye(3) * another_matrix;
  std::cout << my_first_matrix * another_matrix << "\n";

  return 0;
}
