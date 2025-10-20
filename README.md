# BE_SEM_7_LAs

This repository contains coursework, assignments, mini projects, and datasets for the **7th semester of the Bachelor of Engineering program**.  
The workspace is organized by subject and type of work, supporting **C/C++**, **CUDA**, **Python**, and **data science** workflows.

---

## Folder Structure

### **BCT**
You can find scripts in provided PDF files.

---

### **DAA (Design and Analysis of Algorithms)**

**Assignments_PDFs/**  
- Contains PDFs of assignment scripts.

**Execution/**  
- C source files for classic algorithm problems:
  - `fractional_knapsack.c`: Fractional knapsack implementation.  
  - `knapsack_01.c`: 0/1 knapsack problem.  
  - `nqueens_backtrack.c`: N-Queens problem using backtracking.  

**Mini Project/**  
- Contains a mini project on matrix multiplication and performance analysis:
  - `.gitignore`: Local ignore rules for this subproject.  
  - `matrix_mult_cuda.cu`: CUDA implementation of matrix multiplication.  
  - `matrix_mult_win.c`: Windows C implementation of matrix multiplication.  
  - `perf_plot.py`: Python script for plotting performance results.  
  - `results.csv`: CSV file with performance results.  
  - `run_all.bat`: Batch script to run all experiments.  
  - `backup.txt`: Backup or notes.

---

### **GPA (Graphics and Parallel Algorithms)**

**Assignments/Execution/**  
- CUDA and C source files for matrix and vector operations:
  - `cuda_gl.cu`, `matrix_ops.cu`, `pi_calculation.cu`, `vector_addition.c`: CUDA programs for graphics and parallel computation.  
  - `matrix_ops.exp`, `opencl_hw.c`: Export and OpenCL files.

**Assignments/PDFs/**  
- Assignment instructions and solutions in PDF format.

**CIE2/**  
- Additional CUDA and OpenCL programs for internal evaluation:
  - `cuda_array_addition.cu`, `cuda_array_addition.exp`: CUDA array addition and export.  
  - `opencl_day.cpp`, `opencl_name_day.cpp`: OpenCL programs.

---

### **ML (Machine Learning)**

**Datasets/**  
- CSV files for various machine learning tasks:
  - `CustomerChurn.csv`  
  - `diabetes.csv`  
  - `emails.csv`  
  - `sales_data_sample.csv`  
  - `uber.csv`

**Notebooks/**  
- Jupyter notebooks for lab assignments:
  - `LA_1_UBER_FARE_PREDICTION.ipynb`: Uber fare prediction.  
  - `LA_2_Email_Spam_Classification.ipynb`: Email spam classification.  
  - `LA_3_Customer_Churn_Prediction.ipynb`: Customer churn prediction.  
  - `LA_5_Diabetes_Classification.ipynb`: Diabetes classification.  
  - `LA_6_Sales_Clustering.ipynb`: Sales data clustering.

---

## How to Use

- **C/C++/CUDA files:** Compile using appropriate compilers (`gcc`, `nvcc`, etc.). Input files are provided for testing.  
- **Python scripts and notebooks:** Use **Jupyter Notebook** or **Python 3.x**. Datasets are available in the *Datasets* folder.  
- **Batch scripts:** Run `.bat` files on Windows to automate experiments.

---

## Notes

- All compiled files and outputs are ignored via `.gitignore` to keep the repository clean.  
- Each subject folder is self-contained for easy navigation and usage.  
- Feel free to further customize this README with course details, author information, or specific instructions for each assignment or project.
