#include <iostream>
#include <algorithm> // In order to use max
#include <cassert>
#include <iomanip> // to use setw
#include <cmath>
#include <initializer_list>
#include "Matrix.hpp"
#include "Vector.hpp"



// Constructor for matrix of given size
Matrix::Matrix(int rows, int cols) {
  mSize[0] = rows; mSize[1] = cols;
  mData = new double* [rows];

  for (int i = 0; i < rows; i++) {
    mData[i] = new double [cols];

    for (int j = 0; j < cols; j++) {
      mData[i][j] = 0.0;
    }
  }
}

// Constructor for matrix built from list
Matrix::Matrix(int rows, int cols, std::initializer_list<double> input) {
  mSize[0] = rows; mSize[1] = cols;
  mData = new double* [rows];
  int k = 0;

  for (int i = 0; i < rows; i++) {
    mData[i] = new double [cols];

    for (int j = 0; j < cols; j++) {
      mData[i][j] = input.begin()[k];
      k++;
    }
  }
}


// Copy constructor
Matrix::Matrix(const Matrix& mat) {
  int rows, cols;
  rows = mat.mSize[0]; cols = mat.mSize[1];

  mSize[0] = rows; mSize[1] = cols;
  mData = new double* [rows];

  for (int i = 0; i < rows; i++) {
    mData[i] = new double [cols];

    for (int j = 0; j < cols; j++) {
      mData[i][j] = mat.mData[i][j];
    }
  }
}


 // Destructor
 Matrix::~Matrix() {
   int rows = mSize[0];
   for (int i = 0; i < rows; i++) {
     delete[] mData[i];
   }
   delete[] mData;
 }


std::ostream& operator<<(std::ostream& output, const Matrix& mat) {
  int rows = mat.mSize[0];
  int cols = mat.mSize[1];

  // char buffer [50];

  output << "(";
  for (int i = 0; i < rows; i++) {



    for (int j = 0; j < cols; j++) {
      // output << "\t" << mat.mData[i][j];
      if (i == 0 && j == 0) {
        output << std::setw(9) << mat.mData[i][j];
      } else {
        output << std::setw(10) << mat.mData[i][j];
      }
      // sprintf(buffer, "%f", mat.mData[i][j]);

      if (j == cols - 1 && i != rows - 1) {
        output << "\n";
      }
    }

    if (i == rows - 1) {
      output << "\t ) \n";
    }

  }

  return output;
}


double& Matrix::operator()(int i, int j) {
  int rows = mSize[0]; int cols = mSize[1];

  if (i < 1 || i > rows || j < 1 || j > cols){
    // throw Exception("Out of range",
    // "Accessing matrix through (), index out of range");
    std::cerr<<"Index out of range";
  }

  return mData[i-1][j-1];
}


Matrix& Matrix::operator=(const Matrix& mat) {
  assert(mSize[0] == mat.mSize[0] && mSize[1] == mat.mSize[1]);
  // Manage different sizes

  for (int i = 0; i < mSize[0]; i++) {
    for (int j = 0; j < mSize[1]; j++) {
      mData[i][j] = mat.mData[i][j];
    }
  }

  return *this;
}





// Binary operators
Matrix operator+(const Matrix& mat1, const Matrix& mat2) {
  int max_row, max_col;
  max_row = std::max(size(mat1)[0], size(mat2)[0]);
  max_col = std::max(size(mat1)[1], size(mat2)[1]);

  Matrix out(max_row, max_col);

  for (int i = 0; i < size(mat1)[0]; i++) {
    for (int j = 0; j < size(mat1)[1]; j++) {
      out.mData[i][j] += mat1.mData[i][j];
      // We can use (i,j) notation on the left for out but not on the right for
      // mat1 because argument is const and method (i, j) is not declared as const
      // Result is that we cannot unfriend this method
    }
  }

  for (int i = 0; i < size(mat2)[0]; i++) {
    for (int j = 0; j < size(mat2)[1]; j++) {
      out.mData[i][j] += mat2.mData[i][j];
    }
  }

  if (mat1.mSize[0] < max_row || mat2.mSize[0] < max_row
      || mat1.mSize[1] < max_col || mat2.mSize[1] < max_col) {
    std::cerr << "Matrix add - matrices of different sizes \n";
    std::cerr << "Extra entries of smaller matrix assumed to be 0.0\n";
  }

  return out;
}

Matrix operator-(const Matrix& mat) {
  Matrix out(mat);
  for (int i = 0; i < size(mat)[0]; i++) {
    for (int j = 0; j < mat.mSize[1]; j++) {
      out.mData[i][j] = -mat.mData[i][j];
    }
  }
  return out;
}

Matrix operator-(const Matrix& mat1, const Matrix& mat2) {
  Matrix minus_mat2(mat2);
  minus_mat2 = -mat2;

  return mat1 + minus_mat2;
}

Matrix operator*(const Matrix& mat, const double& a) {
  Matrix out(mat);

  for (int i = 0; i < size(mat)[0]; i++) {
    for (int j = 0; j < size(mat)[1]; j++) {
      out.mData[i][j] = a * mat.mData[i][j];
    }
  }

  return out;
}

Matrix operator*(const double& a, const Matrix& mat) {
  return mat * a;
}

Matrix operator/(const Matrix& mat, const double& a) {
  return mat * (1.0/a);
}

Vector operator*(const Matrix& mat, const Vector& vec) {
  int rows = size(mat)[0]; int cols = size(mat)[1];
  assert(cols == length(vec));

  Vector out(rows); // This initializes zero vector

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {

      out.mData[i] += mat.mData[i][j] * vec.mData[j];
    }
  }

  return out;
}

Matrix operator*(const Matrix& matA, const Matrix& matB) {
  int rowsA = size(matA)[0]; int colsA = size(matA)[1];
  int rowsB = size(matB)[0]; int colsB = size(matB)[1];

  assert(colsA == rowsB);
  Matrix out(rowsA, colsB);

  for (int i = 0; i < rowsA; i++) {
    for (int j = 0; j < colsB; j++){
      for (int k = 0; k < colsA; k++) {

        out.mData[i][j] += matA.mData[i][k] * matB.mData[k][j];
      }
    }
  }
  return out;
}

// Matrix& Matrix::operator+=(const Matrix& rhs_mat) {
//   // Need to check question of size
//   // assert(...)
//   for (int i = 0; i < size(rhs_mat)[0]; i++) {
//     for (int j = 0; j < size(rhs_mat)[1]; j++) {
//       this->mData[i][j] = this->mData[i][j] + rhs_mat.mData[i][j];
//     }
//   }
//   return *this;
// }

double norm(const Matrix& mat, int p) {
  double out = 0;
  for (int i = 0; i < mat.mSize[0]; i++) {
    for (int j = 0; j < mat.mSize[1]; j++) {
      out += pow(mat.mData[i][j], p);
    }
  }
  return pow(out, (1.0/p));
}

Matrix transpose(const Matrix& mat) {
  Matrix out(mat);

  for (int i = 0; i < mat.mSize[0]; i++) {
    for (int j = 0; j < mat.mSize[1]; j++) {

      out.mData[i][j] = mat.mData[j][i];
    }
  }
  return out;
}

int* size(const Matrix& mat) {
  return (int*) mat.mSize; //mat.mSize is int[2], (int*) is type casting
}

Matrix eye(int n) {
  Matrix out(n, n);
  for (int i = 1; i <= n; i++) {
    out(i, i) = 1.0;
  }
  return out;
}

// Matrix kron(const Matrix& matA, const Matrix& matB) {
//     int rowsA = size(matA)[0]; int colsA = size(matA)[1];
//     int rowsB = size(matB)[0]; int colsB = size(matB)[1];
//
//     Matrix out(rowsA * rowsB, colsA * colsB);
//     double A_entry, B_entry;
//
//     for (int i = 0; i < rowsA; i++) {
//         for (int j = 0; j < colsA; j++) {
//             A_entry = matA.mData[i][j];
//             // std::cout << i << j << "\t"<< A_entry << "\n";
//             if (A_entry == 0.0) {
//                 continue;
//             } else {
//                 for (int ii = 0; ii < rowsB; ii++) {
//                     for (int jj = 0; jj < colsB; jj++) {
//                       B_entry = matB.mData[ii][jj];
//                       // std::cout << A_entry * matB.mData[ii][jj];
//                         out.mData[ii + i*rowsA][jj + j*colsA] = A_entry * B_entry;
//                         // out.mData[ii][jj] = A_entry * matB.mData[ii][jj];
//                     }
//                 }
//                 // out.mData[i][j] = A_entry * matB.mData[i%rowsB][j%colsB];
//             }
//         }
//     }
//     return out;
// }

Matrix diag(const Vector& vec) {
  int n = length(vec);
  Matrix out(n, n);
  for (int i = 0; i < n; i++) {
    out.mData[i][i] = vec.mData[i];
  }
  return out;
}

Matrix triu(const Matrix& mat) {
  int rows = size(mat)[0]; int cols = size(mat)[1];
  Matrix out(rows, cols);
  for (int j = 0; j < cols; j++) {
    for (int i = 0; i <= j; i++) {
      out.mData[i][j] = mat.mData[i][j];
    }
  }
  return out;
}

Matrix tril(const Matrix& mat) {
  int rows = size(mat)[0]; int cols = size(mat)[1];
  Matrix out(rows, cols);
  for (int i = 0; i < cols; i++) {
    for (int j = 0; j <= i; j++) {
      out.mData[i][j] = mat.mData[i][j];
    }
  }
  return out;
}

Matrix zeros(int rows, int cols) {
  Matrix out(rows, cols);
  return out;
}

Matrix ones(int rows, int cols) {
  Matrix out(rows, cols);
  for (int i = 1; i <= rows; i++) {
    for (int j = 1; j <= cols; j++) {
      out(i,j) = 1.0;
    }
  }
  return out;
}



Vector cgs(const Matrix& A, const Vector& b, const Vector& x0, double tol) {
  int n = length(b);
  assert(size(A)[0] == n && size(A)[0] == size(A)[1]);
  Vector out(n);
  int iter = 0;

  Vector r = b - A*x0;
  Vector p(r);
  Vector Ap(n); // There is no default constructor
  double norm_r_old = r * r, norm_r_new, alpha, beta;

  for (int k = 0; k < n; k++) {
    Ap = A * p;
    alpha = norm_r_old / (p * Ap); // This is inner product so don't need traspose of p
    out = out + alpha * p;
    r = r - alpha * Ap;
    norm_r_new = r * r;

    if (sqrt(norm_r_new) < tol) {
      break;
    }

    beta = norm_r_new/norm_r_old;
    p = r + beta * p;
    norm_r_old = norm_r_new;
    iter++;
  }
  std::cout << "CGS converged in " << iter << " iterations \n";
  return out;
}

Vector jacobi(const Matrix& A, const Vector& b, const Vector& x0, double tol, int MAXITER) {
  int n = length(b);
  assert(size(A)[0] == n && size(A)[0] == size(A)[1]);
  Vector out(n);
  Vector previous(out);
  int iter = 0;

  Vector r = b - A*x0;
  double residue = norm(r);

  // Matrix M = diag(diag(A));
  // Matrix N = M - A;
  // Vector rhs = N*x0 + b;

  while (residue > tol && iter < MAXITER) {
    // out = rhs/M;
    // rhs = N*out + b;
    for (int i = 0; i < n; i++) {
      double tmp = 0.0;
      for (int j = 0; j < n; j++) {
        if (j == i) continue;
        tmp += A.mData[i][j] * previous.mData[j];
      }
      out.mData[i] = (b.mData[i] - tmp)/A.mData[i][i];
    }
    previous = out;

    r = b - A*out;
    residue = norm(r);
    iter++;
  }
  // std::cout << "Jacobi converged in " << iter << " iterations \n";
  return out;
}

Vector gaussSeidel(const Matrix& A, const Vector& b, const Vector& x0, double tol, int MAXITER) {
  int n = length(b);
  assert(size(A)[0] == n && size(A)[0] == size(A)[1]);
  Vector out(n);
  Vector previous(out);
  int iter = 0;

  Vector r = b - A*x0;
  double residue = norm(r);

  // Matrix M = tril(A);
  // Matrix N = M - A;
  // Vector rhs = N*x0 + b;

  while (residue > tol && iter < MAXITER) {
    // out = rhs/M;
    // rhs = N*out + b;
    for (int i = 0; i < n; i++) {
      double tmp = 0.0;
      for (int j = 0; j < n; j++) {
        if (j < i) {
          tmp += A.mData[i][j] * out.mData[j];
        } else if (j > i) {
          tmp += A.mData[i][j] * previous.mData[j];
        }
      }
      out.mData[i] = (b.mData[i] - tmp)/A.mData[i][i];
    }
    previous = out;

    r = b - A*out;
    residue = norm(r);
    iter++;
  }
  // std::cout << "Gauss-Seidel converged in " << iter << " iterations \n";
  return out;
}

Vector sor(const Matrix& A, const Vector& b, const Vector& x0, double omega, double tol, int MAXITER) {
  int n = length(b);
  assert(size(A)[0] == n && size(A)[0] == size(A)[1]);
  Vector out(n);
  Vector previous(out);
  int iter = 0;

  Vector r = b - A*x0;
  double residue = norm(r);

  // Matrix D = diag(diag(A));
  // Matrix L = tril(A) - D;
  // Matrix U = A - L - D;
  //
  // Matrix lhs = D + omega*L;
  // Vector rhs = omega*b + ( (1-omega)*D - omega*U )*x0 ;

  while (residue > tol && iter < MAXITER) {
    // out = rhs/lhs;
    // rhs = omega*b + ( (1-omega)*D - omega*U )*out;
    for (int i = 0; i < n; i++) {
      double tmp = 0.0;
      for (int j = 0; j < n; j++) {
        if (j < i) {
          tmp += A.mData[i][j] * out.mData[j];
        } else if (j > i) {
          tmp += A.mData[i][j] * previous.mData[j];
        }
      }
      out.mData[i] = omega * (b.mData[i] - tmp)/A.mData[i][i] + (1-omega) * previous.mData[i];
    }
    previous = out;

    r = b - A*out;
    residue = norm(r);
    iter++;
  }
  // std::cout << "SOR converged in " << iter << " iterations \n";
  return out;
}


Matrix laplacian(int mesh_size) {
  int squared = pow(mesh_size,2);
  Matrix out(squared, squared);

  for (int i = 2; i < squared; i++) {
    if (i-mesh_size > 0) {
      out(i, i-mesh_size) = -1.0;
    }
    out(i, i-1) = -1.0;
    out(i, i) = 4.0;
    out(i, i+1) = -1.0;
    if (i+mesh_size <= squared) {
      out(i, i+mesh_size) = -1.0;
    }
  }

  out(1, 1) = 4.0;
  out(1, 2) = -1.0;
  out(1, mesh_size+1) = -1.0;

  out(squared, squared-mesh_size) = -1.0;
  out(squared, squared-1) = -1.0;
  out(squared, squared) = 4.0;

  return out;
}
