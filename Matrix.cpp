#include <iostream>
#include <algorithm> // In order to use max
#include <cassert>
#include <iomanip> // to use setw
#include <cmath>
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
    throw Exception("Out of range",
    "Accessing matrix through (), index out of range");
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
  max_row = std::max(mat1.mSize[0], mat2.mSize[0]);
  max_col = std::max(mat1.mSize[1], mat2.mSize[1]);

  Matrix out(max_row, max_col);

  for (int i = 0; i < mat1.mSize[0]; i++) {
    for (int j = 0; j < mat1.mSize[1]; j++) {
      out.mData[i][j] += mat1.mData[i][j];
    }
  }

  for (int i = 0; i < mat2.mSize[0]; i++) {
    for (int j = 0; j < mat2.mSize[1]; j++) {
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
  for (int i = 0; i < mat.mSize[0]; i++) {
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

  for (int i = 0; i < mat.mSize[0]; i++) {
    for (int j = 0; j < mat.mSize[1]; j++) {
      out.mData[i][j] = a*mat.mData[i][j];
    }
  }

  return out;
}

Matrix operator*(const double& a, const Matrix& mat) {
  return mat*a;
}

Matrix operator/(const Matrix& mat, const double& a) {
  return mat*(1/a);
}

Vector operator*(const Matrix& mat, const Vector& vec) {
  int rows = mat.mSize[0]; int cols = mat.mSize[1];
  assert(cols == vec.mSize);

  Vector::Vector out(rows); // This initializes zero vector

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {

      out.mData[i] += mat.mData[i][j] * vec.mData[j];
    }
  }

  return out;
}

Matrix operator*(const Matrix& matA, const Matrix& matB) {
  int rowsA = matA.mSize[0]; int colsA = matA.mSize[1];
  int rowsB = matB.mSize[0]; int colsB = matB.mSize[1];

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



Matrix transpose(const Matrix& mat) {
  Matrix out(mat);

  for (int i = 0; i < mat.mSize[0]; i++) {
    for (int j = 0; j < mat.mSize[1]; j++) {

      out.mData[i][j] = mat.mData[j][i];
    }
  }
  return out;
}

int* size(const Matrix& mat){
  return (int*) mat.mSize; //mat.mSize is int[2], (int*) is type casting
}

Matrix eye(int n) {
  Matrix out(n, n);
  for (int i = 0; i < n; i++) {
    out.mData[i][i] = 1.0;
  }
  return out;
}

Matrix diag(const Vector& vec) {
  int n = length(vec);
  Matrix out(n, n);
  for (int i = 0; i < n; i++) {
    out.mData[i][i] = vec.mData[i];
  }
  return out;
}


/* ----------------- SymmetricMatrix ----------------- */

PositiveDefiniteMatrix::PositiveDefiniteMatrix(int size) : Matrix (size, size) {
 for (int i = 0; i < size; i++) {
   mData[i][i] = 1.0;
 }
}

Vector cgs(const SymmPosDefMatrix& A, const Vector& b, const Vector& x0, double tol) {
  int n = length(b);
  assert(size(A)[0] == n);
  Vector out(n);

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
  }
  return out;
}
