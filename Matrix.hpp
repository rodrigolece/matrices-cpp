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
  Matrix(int rows, int cols, std::initializer_list<double> input); // construct matrix from list

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
  // friend Matrix operator-(const Matrix& mat1, const Matrix& mat2); // friend not necessary
  friend Matrix operator*(const Matrix& mat, const double& a);
  // friend Matrix operator*(const double& a, const Matrix& mat);
  // friend Matrix operator/(const Matrix& mat, const double& a);
  friend Vector operator*(const Matrix& mat, const Vector& vec);
  friend Matrix operator*(const Matrix& matA, const Matrix& matB);
  friend Vector operator/(const Vector& vec, const Matrix& mat);

  // Other useful functions
  friend double norm(const Matrix& mat, int p);
  friend Matrix transpose(const Matrix& mat);
  friend int* size(const Matrix& mat);
  friend Matrix diag(const Vector& vec);
  friend Vector diag(const Matrix& mat); // <---- check if unfriend is possible
  friend Matrix triu(const Matrix& mat);
  friend Matrix tril(const Matrix& mat);

  friend Vector jacobi(const Matrix& A, const Vector& b, const Vector& x0, double tol, int MAXITER);
  friend Vector gaussSeidel(const Matrix& A, const Vector& b, const Vector& x0, double tol, int MAXITER);
  friend Vector sor(const Matrix& A, const Vector& b, const Vector& x0, double omega, double tol, int MAXITER);

  friend Matrix laplacian(int mesh_size);
};

Matrix operator+(const Matrix& mat1, const Matrix& mat2);
Matrix operator-(const Matrix& mat);
Matrix operator-(const Matrix& mat1, const Matrix& mat2);
Matrix operator*(const Matrix& mat, const double& a);
Matrix operator*(const double& a, const Matrix& mat);
Matrix operator/(const Matrix& mat, const double& a);
Vector operator*(const Matrix& mat, const Vector& vec);
Matrix operator*(const Matrix& matA, const Matrix& matB);

double norm(const Matrix& mat, int p = 2); // by default frobenius norm
Matrix transpose(const Matrix& mat);
int* size(const Matrix& mat);
Matrix eye(int n);
Matrix diag(const Vector& vec);
Matrix triu(const Matrix& mat);
Matrix tril(const Matrix& mat);
Matrix zeros(int rows, int cols);
Matrix ones(int rows, int cols);

Vector cgs(const Matrix& A, const Vector& b, const Vector& x0, double tol = 1e-10);
Vector jacobi(const Matrix& A, const Vector& b, const Vector& x0, double tol = 1e-6,
              int MAXITER = 100);
Vector gaussSeidel(const Matrix& A, const Vector& b, const Vector& x0,
                   double tol = 1e-6, int MAXITER = 100);
Vector sor(const Matrix& A, const Vector& b, const Vector& x0, double omega,
           double tol = 1e-6, int MAXITER = 100);

Matrix laplacian(int mesh_size);
Vector manufacturedSourceTerm(int mesh_size);
Vector manufacturedSolution(int mesh_size);
Vector sourceTerm(int mesh_size);
#endif
