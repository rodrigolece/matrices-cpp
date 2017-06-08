#ifndef MATRIXDEF
#define MATRIXDEF



//  ***********************
//  *  Class of Matrices  *
//  ***********************




#include <cmath>
#include "Exception.hpp"//  This class throws errors using the class "error""

class Vector; // Forward declaration

class Matrix {
private:
  double** mData;
  int mSize[2];

public:
  Matrix(const Matrix& mat); // copy constructor
  Matrix(int rows, int cols); // zeros matrix of given size

  // destructor
  ~Matrix();

  // Chevron operator for display
  friend std::ostream& operator<<(std::ostream& output, const Matrix& mat);
  // No prototype for chevron operator ouside because already defined in ostream

  // Read/assign operator like Matlab's
  double& operator()(int i, int j);
  // Assign operator
  Matrix& operator=(const Matrix& mat);

  // Binary operators
  friend Matrix operator+(const Matrix& mat1, const Matrix& mat2);
  friend Matrix operator-(const Matrix& mat);
  friend Matrix operator-(const Matrix& mat1, const Matrix& mat2);
  friend Matrix operator*(const Matrix& mat, const double& a);
  friend Matrix operator*(const double& a, const Matrix& mat);
  friend Matrix operator/(const Matrix& mat, const double& a);
  friend Vector operator*(const Matrix& mat, const Vector& vec);
  friend Matrix operator*(const Matrix& matA, const Matrix& matB);
  friend Vector operator/(const Vector& vec, const Matrix& mat);

  // Other useful functions
  friend Matrix transpose(const Matrix& mat);
  friend int* size(const Matrix& mat);
  friend Matrix eye(int n);
  friend Matrix diag(const Vector& vec);
  friend Vector diag(const Matrix& mat);
};

Matrix operator+(const Matrix& mat1, const Matrix& mat2);
Matrix operator-(const Matrix& mat);
Matrix operator-(const Matrix& mat1, const Matrix& mat2);
Matrix operator*(const Matrix& mat, const double& a);
Matrix operator*(const double& a, const Matrix& mat);
Matrix operator/(const Matrix& mat, const double& a);
Vector operator*(const Matrix& mat, const Vector& vec);
Matrix operator*(const Matrix& matA, const Matrix& matB);

Matrix transpose(const Matrix& mat);
int* size(const Matrix& mat);
Matrix eye(int n);
Matrix diag(const Vector& vec);

#endif
