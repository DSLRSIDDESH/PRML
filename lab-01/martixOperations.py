#CS21B2019 DSLR SIDDESH
import numpy as np

def matrix(num):
    m, n = map(int, input(f"Enter the M,N of Matrix-{num} : ").split())
    print(f"Enter the elements of Matrix-{num}")
    matrix = []
    for _ in range(m):
        r_i = list(map(int, input().split()))
        matrix.append(r_i) if len(r_i) == n else print("Invalid input size!!")
    mat = np.array(matrix)
    return mat

def invalid():
  print("Invalid size of Matrix-2 for the operation")

mat1 = matrix(1)
print("Matrix-1 :")
print(mat1)
mat2 = matrix(2)
print("Matrix-2 :")
print(mat2)

# determining which operations are possible
opr_list = [3, 6, 12]
if mat1.shape == mat2.shape:
  opr_list.extend([1, 2, 4])
if mat1.shape[1] == mat2.shape[0]:
  opr_list.append(5)
if mat1.shape[0] == mat1.shape[1]:
  opr_list.extend([7, 8, 9, 10, 11])

print("1.Matrix Addition")
print("2.Matrix Subtraction")
print("3.Scalar Matrix Multiplication")
print("4.Elementwise Matrix Multiplication")
print("5.Matrix Multiplication")
print("6.Matrix Transpose")
print("7.Trace of a Matrix")
print("8.Solve System of Linear Equations")
print("9.Determinant")
print("10.Inverse")
print("11.Eigen Value and Eigen Vector")
print("12.Exit")

while True:
  choice = int(input("\nEnter your choice : "))
  if choice in opr_list:
    if choice == 1:
        print("Addition of Matrices is :")
        print(mat1 + mat2)
    elif choice == 2:
        print("Subtraction of Matrices is :")
        print(mat1 - mat2)
    elif choice == 3:
        sc = int(input("Enter the scalar : "))
        print("Scalar Multiplication of Matrix is :")
        print(mat1 * sc)
    elif choice == 4:
        print("Elementwise Matrix Multiplication is :")
        print(mat1 * mat2)
    elif choice == 5:
        print("Matrix Multiplication is :")
        print(np.matmul(mat1, mat2))
    elif choice == 6:
        print("Transpose of Matrix is :")
        print(np.transpose(mat1))
    elif choice == 7:
        print("Trace of Matrix is :", np.trace(mat1))
    elif choice == 8:
        b = list(map(int, input("Enter the elements of vector B : ").split()))
        if len(b) == mat1.shape[0]:
            print("Solution vector : ",np.linalg.solve(mat1, b))
        else:
            print("Invalid size for B!!")
    elif choice == 9:
        print("Determinant of Matrix is :",np.linalg.det(mat1).round())
    elif choice == 10:
        print("Inverse of Matrix is :")
        print(np.linalg.inv(mat1))
    elif choice == 11:
        output = np.linalg.eig(mat1)
        print("Eigen values of matrix :", output[0])
        print("Eigen vectors of matrix :")
        print(output[1])
    elif choice == 12:
        print("Exiting..")
        break
    else:
        print("Invalid input!!")
  else:
    invalid()