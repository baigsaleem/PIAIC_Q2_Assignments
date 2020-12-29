{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "NgUeapcvmIRh"
   },
   "source": [
    "# **Assignment For Numpy**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "yhr3JicwnA-4"
   },
   "source": [
    "Difficulty Level **Beginner**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "hVPqDJ1SnKSh"
   },
   "source": [
    "1. Import the numpy package under the name np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "id": "SePu31zKmgS-"
   },
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "d0gO81krnL7g"
   },
   "source": [
    "2. Create a null vector of size 10 "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {
    "id": "2s6o2JPgnSBr"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.zeros(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "BJTMK03fnS1w"
   },
   "source": [
    "3. Create a vector with values ranging from 10 to 49"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "id": "qRXxXuWdnj1M"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,\n",
       "       27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,\n",
       "       44, 45, 46, 47, 48, 49])"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "vec1 = np.arange(10,50)\n",
    "vec1"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "n63lzg5Onkcn"
   },
   "source": [
    "4. Find the shape of previous array in question 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "id": "1FueBdqPn_q7"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(40,)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.shape(vec1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "AzMlQj1MoAVe"
   },
   "source": [
    "5. Print the type of the previous array in question 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "id": "AO9PYmdWoE1L"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dtype('int64')"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "vec1.dtype"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "0qaKS13Yon-U"
   },
   "source": [
    "6. Print the numpy version and the configuration\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "id": "T3wY24e1oorh"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1.19.2\n",
      "blas_mkl_info:\n",
      "    libraries = ['mkl_rt', 'pthread']\n",
      "    library_dirs = ['/home/salemo/anaconda3/lib']\n",
      "    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]\n",
      "    include_dirs = ['/home/salemo/anaconda3/include']\n",
      "blas_opt_info:\n",
      "    libraries = ['mkl_rt', 'pthread']\n",
      "    library_dirs = ['/home/salemo/anaconda3/lib']\n",
      "    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]\n",
      "    include_dirs = ['/home/salemo/anaconda3/include']\n",
      "lapack_mkl_info:\n",
      "    libraries = ['mkl_rt', 'pthread']\n",
      "    library_dirs = ['/home/salemo/anaconda3/lib']\n",
      "    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]\n",
      "    include_dirs = ['/home/salemo/anaconda3/include']\n",
      "lapack_opt_info:\n",
      "    libraries = ['mkl_rt', 'pthread']\n",
      "    library_dirs = ['/home/salemo/anaconda3/lib']\n",
      "    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]\n",
      "    include_dirs = ['/home/salemo/anaconda3/include']\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "print(np.__version__)\n",
    "print(np.show_config())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ZyUUiQf5oyvE"
   },
   "source": [
    "7. Print the dimension of the array in question 3\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "id": "pLXfuIqIo0vq"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "vec1.ndim"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "JYVMuFrqpBdV"
   },
   "source": [
    "8. Create a boolean array with all the True values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "id": "3apZsISzpFKR"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True,  True,  True,  True,  True,  True,\n",
       "        True,  True,  True,  True])"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bol_vec = np.array(vec1>0)\n",
    "bol_vec"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "4zbBooWZpPBU"
   },
   "source": [
    "9. Create a two dimensional array\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "id": "KfPQEiVIpdTo"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0, 1, 2, 3],\n",
       "       [4, 5, 6, 7]])"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "vec2d = np.array([[0,1,2,3],[4,5,6,7]])\n",
    "vec2d"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "do9wPAFbpqC7"
   },
   "source": [
    "10. Create a three dimensional array\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "id": "ZCO_q-KZqAeW"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[ 1,  2,  3,  4,  5],\n",
       "        [ 6,  7,  8,  9, 10],\n",
       "        [11, 12, 13, 14, 15]]])"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "vec3d = np.array([[[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]]])\n",
    "vec3d"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "a4iysJ87qEeb"
   },
   "source": [
    "Difficulty Level **Easy**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "DehgqszSqjY6"
   },
   "source": [
    "11. Reverse a vector (first element becomes last)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "id": "tfevmc4VqkWu"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 2 3 4 5]\n",
      "[5 4 3 2 1]\n"
     ]
    }
   ],
   "source": [
    "vec_notFliped = np.arange(1,6)\n",
    "print(vec_notFliped)\n",
    "vec_fliped = np.flip(vec_notFliped)\n",
    "print(vec_fliped)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "yoATXS-qqk_q"
   },
   "source": [
    "12. Create a null vector of size 10 but the fifth value which is 1 "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {
    "id": "AheZ32Owqpte"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.])"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "vec_null = np.zeros(10)\n",
    "vec_null[5] = 1\n",
    "vec_null"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "RZRzOBbFsY0w"
   },
   "source": [
    "13. Create a 3x3 identity matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {
    "id": "va2ou0BHsvEj"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[0, 1, 2],\n",
       "       [3, 4, 5],\n",
       "       [6, 7, 8]])"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr3 = np.arange(9).reshape(3,3)\n",
    "arr3"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "lnN5drkUs6o2"
   },
   "source": [
    "14. arr = np.array([1, 2, 3, 4, 5]) \n",
    "\n",
    "---\n",
    "\n",
    " Convert the data type of the given array from int to float "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {
    "id": "59EIQt6otKUB"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1., 2., 3., 4., 5.])"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr = np.array([1,2,3,4,5],dtype=float)\n",
    "arr"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "fSL2AKJetTes"
   },
   "source": [
    "15. arr1 =          np.array([[1., 2., 3.],\n",
    "\n",
    "                    [4., 5., 6.]])  \n",
    "                      \n",
    "    arr2 = np.array([[0., 4., 1.],\n",
    "     \n",
    "                   [7., 2., 12.]])\n",
    "\n",
    "---\n",
    "\n",
    "\n",
    "Multiply arr1 with arr2\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {
    "id": "7VDXgRQDuJNV"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0.,  8.,  3.],\n",
       "       [28., 10., 72.]])"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr1 = np.array([[1., 2., 3.],[4., 5., 6.]])\n",
    "arr2 = np.array([[0., 4., 1.],[7., 2., 12.]])\n",
    "arr1*arr2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "a8yJ438-uKni"
   },
   "source": [
    "16. arr1 = np.array([[1., 2., 3.],\n",
    "                    [4., 5., 6.]]) \n",
    "                    \n",
    "    arr2 = np.array([[0., 4., 1.], \n",
    "                    [7., 2., 12.]])\n",
    "\n",
    "\n",
    "---\n",
    "\n",
    "Make an array by comparing both the arrays provided above"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {
    "id": "3MPDaAlYueF3"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[False, False, False],\n",
       "       [False, False, False]])"
      ]
     },
     "execution_count": 56,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr1 = np.array([[1., 2., 3.],[4., 5., 6.]])\n",
    "arr2 = np.array([[0., 4., 1.],[7., 2., 12.]])\n",
    "arr1==arr2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "3xiD0eJluewk"
   },
   "source": [
    "17. Extract all odd numbers from arr with values(0-9)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {
    "id": "J_qbt_EuvM7l"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 3, 5, 7, 9])"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr = np.arange(10)\n",
    "arr_odd = arr[1:10:2]\n",
    "arr_odd"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "zpg5cyLsvPoZ"
   },
   "source": [
    "18. Replace all odd numbers to -1 from previous array"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {
    "id": "rXUV1-ULvQdd"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([-1, -1, -1, -1, -1])"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr_odd[:]=-1\n",
    "arr_odd"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "FWpHTWTDvjvi"
   },
   "source": [
    "19. arr = np.arange(10)\n",
    "\n",
    "\n",
    "---\n",
    "\n",
    "Replace the values of indexes 5,6,7 and 8 to **12**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {
    "id": "ensZ3lqwvlSb"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 0,  1,  2,  3,  4, 12, 12, 12, 12,  9])"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr = np.arange(10)\n",
    "arr[5:9]=12\n",
    "arr"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "ib-vBffAv0zy"
   },
   "source": [
    "20. Create a 2d array with 1 on the border and 0 inside"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {
    "id": "RmzcjNYrwDrM"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1., 1., 1., 1., 1., 1., 1.],\n",
       "       [1., 0., 0., 0., 0., 0., 1.],\n",
       "       [1., 0., 0., 0., 0., 0., 1.],\n",
       "       [1., 0., 0., 0., 0., 0., 1.],\n",
       "       [1., 0., 0., 0., 0., 0., 1.],\n",
       "       [1., 0., 0., 0., 0., 0., 1.],\n",
       "       [1., 1., 1., 1., 1., 1., 1.]])"
      ]
     },
     "execution_count": 72,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr_ones = np.ones((7,7))\n",
    "arr_ones[1:-1,1:-1]=0\n",
    "arr_ones"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "E1diIFSEwEmh"
   },
   "source": [
    "Difficulty Level **Medium**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "eGLZmNb_wKb4"
   },
   "source": [
    "21. arr2d = np.array([[1, 2, 3],\n",
    "\n",
    "                    [4, 5, 6], \n",
    "\n",
    "                    [7, 8, 9]])\n",
    "\n",
    "---\n",
    "\n",
    "Replace the value 5 to 12"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {
    "id": "VdUnsUOZwJFz"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 1,  2,  3],\n",
       "       [ 4, 12,  6],\n",
       "       [ 7,  8,  9]])"
      ]
     },
     "execution_count": 74,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr2d = np.array([[1, 2, 3],[4, 5, 6], [7, 8, 9]])\n",
    "arr2d[1][1]=12\n",
    "arr2d"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "TEbFhlPZxaKO"
   },
   "source": [
    "22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])\n",
    "\n",
    "---\n",
    "Convert all the values of 1st array to 64\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {
    "id": "eYzephU8xmsg"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[64, 64, 64],\n",
       "        [64, 64, 64]],\n",
       "\n",
       "       [[ 7,  8,  9],\n",
       "        [10, 11, 12]]])"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])\n",
    "arr3d[0][:]=64\n",
    "arr3d"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "vZj0Ndvnx6f0"
   },
   "source": [
    "23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {
    "id": "pM4e9YfdyHSv"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 2, 3, 4])"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr_2d = np.arange(10).reshape(2,5)\n",
    "arr_2d[0,:]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "C3dkW0lZyVps"
   },
   "source": [
    "24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {
    "id": "IrrzK2xgygqV"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0 1 2 3 4]\n",
      " [5 6 7 8 9]]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "6"
      ]
     },
     "execution_count": 78,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(arr_2d)\n",
    "arr_2d[1,1]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "G3dLwOO9yyrE"
   },
   "source": [
    "25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {
    "id": "5Bx4fqsLyqGD"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0 1 2]\n",
      " [3 4 5]\n",
      " [6 7 8]]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([[2],\n",
       "       [5]])"
      ]
     },
     "execution_count": 86,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "arr2d = np.arange(9).reshape(3,3)\n",
    "print(arr2d)\n",
    "arr2d[0:2,-1:]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "CQ9YST5jy0IL"
   },
   "source": [
    "26. Create a 10x10 array with random values and find the minimum and maximum values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "metadata": {
    "id": "lhJPugjQzG6W"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[-0.17356676 -1.06837931 -0.63921055 -1.47456659  1.11385436 -0.85321066\n",
      "   1.41707166  0.78383808 -0.19083847  0.80691631]\n",
      " [ 0.82730569  0.05775764 -0.24563921  1.84485245  0.03750533 -0.61827841\n",
      "   0.81200097  1.52772798 -0.04570209  0.88679476]\n",
      " [-1.74781545  0.6467892   1.24058724 -2.17929656  0.85945841  0.56611144\n",
      "   1.6730275   1.12121012 -0.13101022 -0.12329078]\n",
      " [ 1.34205824  0.99845328 -0.74731994  0.57025202  0.68762366  0.06437658\n",
      "  -0.38712691  0.96353658 -0.83784245 -0.2896742 ]\n",
      " [ 0.01141723  1.84859658 -0.34151602  0.3515877  -0.99580777  0.26230376\n",
      "  -0.9676026   1.28613477 -1.80611751 -1.36076278]\n",
      " [-0.62094114  0.13880315 -2.05573444  1.07662783 -0.60051237 -0.45272754\n",
      "   1.04736692  1.35183421 -0.67173215 -0.65700394]\n",
      " [-0.882152    1.46768867 -0.35858745 -0.09758142  0.3040106   1.14977707\n",
      "   0.0492953   0.49596534  0.15229895 -0.74071027]\n",
      " [-1.1152904  -1.19168694 -0.49246697  0.40963466  0.30733091  0.16846324\n",
      "   1.38747629 -1.21652382 -0.01988218 -0.63710116]\n",
      " [-1.26057741 -0.10096627 -1.52831335  0.20605336 -0.07366344  0.22973291\n",
      "   1.22721887 -0.04518145 -0.02673798 -0.63923968]\n",
      " [ 0.19180785 -0.27609053  1.10295905  0.27571279  0.09964802  0.00613657\n",
      "  -0.98232274 -0.09114627  0.32432649  1.6588568 ]]\n",
      "-2.1792965632084464\n",
      "1.8485965830134208\n"
     ]
    }
   ],
   "source": [
    "arr10 = np.random.randn(10,10)\n",
    "print(arr10)\n",
    "print(np.amin(arr10))\n",
    "print(np.amax(arr10))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "cAgDt6KazHk6"
   },
   "source": [
    "27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])\n",
    "---\n",
    "Find the common items between a and b\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {
    "id": "kBXZanxIzs_F"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([2, 4])"
      ]
     },
     "execution_count": 94,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a = np.array([1,2,3,2,3,4,3,4,5,6]) \n",
    "b = np.array([7,2,10,2,7,4,9,4,9,8])\n",
    "np.intersect1d(a,b)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "lF9IzWOAztpD"
   },
   "source": [
    "28. a = np.array([1,2,3,2,3,4,3,4,5,6])\n",
    "b = np.array([7,2,10,2,7,4,9,4,9,8])\n",
    "\n",
    "---\n",
    "Find the positions where elements of a and b match\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "metadata": {
    "id": "JR-mHQLH0HY_"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(array([1, 3, 5, 7]),)"
      ]
     },
     "execution_count": 100,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "a = np.array([1,2,3,2,3,4,3,4,5,6]) \n",
    "b = np.array([7,2,10,2,7,4,9,4,9,8])\n",
    "np.where(a==b)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "mMDiefpH0H1x"
   },
   "source": [
    "29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)\n",
    "\n",
    "---\n",
    "Find all the values from array **data** where the values from array **names** are not equal to **Will**\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 118,
   "metadata": {
    "id": "gfc1Lpf81ZSR"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[-0.66927369  0.3647989   0.66797574  0.58535643]\n",
      " [-0.22048696 -0.79679744  0.43687978  0.86865393]\n",
      " [-0.06883846 -1.63491084  0.44948178 -0.82615203]\n",
      " [-0.53664896 -0.63624569  1.1618478   0.45541899]\n",
      " [-1.92329514  0.15510509 -0.21483944  0.25298279]\n",
      " [-0.10618841  3.00108016 -0.10749933  0.06917207]\n",
      " [ 0.75209257 -1.55069347  1.34521775  0.48372283]]\n",
      "(array([0, 1, 3, 5, 6]),)\n"
     ]
    }
   ],
   "source": [
    "names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) \n",
    "data = np.random.randn(7, 4)\n",
    "print(data)\n",
    "x=np.where(names!='Will')\n",
    "print(x)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "52zEjDMf1aXS"
   },
   "source": [
    "30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)\n",
    "\n",
    "---\n",
    "Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 120,
   "metadata": {
    "id": "8RYG0kLO1mHw"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[-0.57938849 -0.1434739  -0.3616586  -0.05290094]\n",
      " [-0.37898511  0.7158721   1.92560018 -1.34387205]\n",
      " [ 0.90293135  0.08432526 -1.27537064 -0.78206838]\n",
      " [-0.92508915  0.86815599 -0.26130372  0.4333448 ]\n",
      " [-0.05693261  0.9886376  -0.48350871  1.01405898]\n",
      " [-0.976871   -1.22137988  0.99900173 -0.84409734]\n",
      " [-0.26061766 -0.60517905 -0.75044497 -0.7198589 ]]\n",
      "(array([0, 2, 3, 4]),)\n"
     ]
    }
   ],
   "source": [
    "names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) \n",
    "data = np.random.randn(7, 4)\n",
    "print(data)\n",
    "y=np.where(names!=('Will'and'Joe'))\n",
    "print(y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Q7hjf2bd2dCY"
   },
   "source": [
    "Difficulty Level **Hard**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "b48g7aRA2jVM"
   },
   "source": [
    "31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 123,
   "metadata": {
    "id": "EbwfSCrW2f9_"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 1  2  3]\n",
      " [ 4  5  6]\n",
      " [ 7  8  9]\n",
      " [10 11 12]\n",
      " [13 14 15]]\n"
     ]
    }
   ],
   "source": [
    "a = np.arange(1,16).reshape(5,3)\n",
    "print(a)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Ei_nDHtO3qBk"
   },
   "source": [
    "32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "metadata": {
    "id": "phOxVpBM4M6D"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[ 1  2  3  4]\n",
      "  [ 5  6  7  8]]\n",
      "\n",
      " [[ 9 10 11 12]\n",
      "  [13 14 15 16]]]\n"
     ]
    }
   ],
   "source": [
    "b = np.arange(1,17).reshape(2,2,4)\n",
    "print(b)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "uyiQaMjA4d_x"
   },
   "source": [
    "33. Swap axes of the array you created in Question 32"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 139,
   "metadata": {
    "id": "w2m9p8064VtR"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[ 1,  5],\n",
       "        [ 2,  6],\n",
       "        [ 3,  7],\n",
       "        [ 4,  8]],\n",
       "\n",
       "       [[ 9, 13],\n",
       "        [10, 14],\n",
       "        [11, 15],\n",
       "        [12, 16]]])"
      ]
     },
     "execution_count": 139,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "np.swapaxes(b,1,2)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "2dlU_yUR4mVZ"
   },
   "source": [
    "34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "metadata": {
    "id": "tmBabEJ-5JgB"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.         1.         1.41421356 1.73205081 2.         2.23606798\n",
      " 2.44948974 2.64575131 2.82842712 3.        ]\n"
     ]
    }
   ],
   "source": [
    "arr10 = np.sqrt(np.arange(10))\n",
    "arr10[:][np.where(arr10<0.5)]=0\n",
    "print(arr10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "SNAM4dpu5RKA"
   },
   "source": [
    "35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 175,
   "metadata": {
    "id": "3pxebU2b5q9Z"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 0.03690877  1.58063925 -1.42626211 -0.82874419  0.36230251  0.03096061\n",
      " -0.84468395 -0.68513794 -0.50719013 -0.50887787 -0.70071793  1.30528733]\n",
      "[-0.40312536 -0.39783443  0.86298956  1.57972884 -0.37884701  2.98119108\n",
      " -1.18396355  0.31417768  0.92586606  1.32269666 -0.46540401 -0.27736531]\n",
      "1.5806392520525685\n",
      "2.9811910824832193\n",
      "[1.58063925 2.58063925]\n"
     ]
    }
   ],
   "source": [
    "a1 = np.random.randn(12)\n",
    "a2 = np.random.randn(12)\n",
    "print(a1)\n",
    "print(a2)\n",
    "a1_max = np.amax(a1)\n",
    "a2_max = np.amax(a2)\n",
    "print(a1_max)\n",
    "print(a2_max)\n",
    "a3 = np.arange(a1_max,a2_max,1)\n",
    "print(a3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "fz_bpqm28qmD"
   },
   "source": [
    "36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])\n",
    "\n",
    "---\n",
    "Find the unique names and sort them out!\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 177,
   "metadata": {
    "id": "hiSNMgcY87Ey"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Bob' 'Joe' 'Will']\n"
     ]
    }
   ],
   "source": [
    "names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])\n",
    "unique_names = np.unique(names)\n",
    "sorted_names = np.sort(unique_names)\n",
    "print(sorted_names)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "dguMNzob870H"
   },
   "source": [
    "37. a = np.array([1,2,3,4,5])\n",
    "b = np.array([5,6,7,8,9])\n",
    "\n",
    "---\n",
    "From array a remove all items present in array b\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 181,
   "metadata": {
    "id": "eCNJXPvK9IAr"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 2 3 4]\n"
     ]
    }
   ],
   "source": [
    "a = np.array([1,2,3,4,5]) \n",
    "b = np.array([5,6,7,8,9])\n",
    "c = np.setdiff1d(a,b)\n",
    "print(c)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "gz4Dvd4b_Akh"
   },
   "source": [
    "38.  Following is the input NumPy array delete column two and insert following new column in its place.\n",
    "\n",
    "---\n",
    "sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) \n",
    "\n",
    "\n",
    "---\n",
    "\n",
    "newColumn = numpy.array([[10,10,10]])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 206,
   "metadata": {
    "id": "lLWAJWJJ_I3d"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[34, 10, 73],\n",
       "       [82, 10, 12],\n",
       "       [53, 10, 66]])"
      ]
     },
     "execution_count": 206,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])\n",
    "newColumn = np.array([[10,10,10]])\n",
    "new_array = np.delete(sampleArray,1,axis=1)\n",
    "np.insert(new_array,1,newColumn,axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "1lIfmbQt_J_W"
   },
   "source": [
    "39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])\n",
    "\n",
    "\n",
    "---\n",
    "Find the dot product of the above two matrix\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 207,
   "metadata": {
    "id": "fV2-vXcR_vmu"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 28.,  64.],\n",
       "       [ 67., 181.]])"
      ]
     },
     "execution_count": 207,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x = np.array([[1., 2., 3.], [4., 5., 6.]]) \n",
    "y = np.array([[6., 23.], [-1, 7], [8, 9]])\n",
    "np.dot(x,y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "5U-4odUw_wP0"
   },
   "source": [
    "40. Generate a matrix of 20 random values and find its cumulative sum"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 210,
   "metadata": {
    "id": "_oOHdiYEAF8N"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[-1.14695184  0.37530961 -0.63541612  0.01207673 -0.92233081 -0.02176586\n",
      "  0.03196734  0.50465316 -0.26142592 -0.37712834  0.14216086 -0.14307117\n",
      "  0.9509759  -1.01565348  0.4730021   1.7317727   0.46434347 -0.65475079\n",
      " -0.68942158 -0.94381963]\n",
      "[-1.14695184 -0.77164223 -1.40705836 -1.39498162 -2.31731243 -2.33907829\n",
      " -2.30711096 -1.8024578  -2.06388371 -2.44101205 -2.29885119 -2.44192236\n",
      " -1.49094646 -2.50659994 -2.03359784 -0.30182514  0.16251833 -0.49223246\n",
      " -1.18165404 -2.12547367]\n"
     ]
    }
   ],
   "source": [
    "matrix_rand = np.random.randn(20)\n",
    "print(matrix_rand)\n",
    "mat_cumsum = np.cumsum(matrix_rand)\n",
    "print(mat_cumsum)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "name": "Numpy-Assignment-PIAIC-Batch35-Q2.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
