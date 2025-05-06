// solver_std_sweep.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <omp.h>
#include <string>

// Grid size
constexpr int NX = 256;
constexpr int NY = 256;

// Time stepping
constexpr double dt = 0.01;
constexpr int   STEPS = 5000;
constexpr int   OUTPUT_INTERVAL = 25;  

// Gauss-Seidel iteration
constexpr int GS_ITER = 20;

inline int idx(int i, int j) { return i + j*NX; }
inline int wrap(int x, int max) {
    if(x >= max) return x - max;
    if(x < 0)    return x + max;
    return x;
}

// Red-Black Gauss-Seidel for (I – dt·D·Laplacian) X = RHS
void diffuseGaussSeidel(std::vector<double> &X,
                        const std::vector<double> &RHS,
                        double diffCoeff) {
    double r = dt * diffCoeff;
    for(int iter=0; iter<GS_ITER; iter++) {
        // Red cells
        #pragma omp parallel for collapse(2)
        for(int j=0; j<NY; j++){
          for(int i=0; i<NX; i++){
            if(((i+j)&1)==0){
              int id = idx(i,j);
              int ip=wrap(i+1,NX), im=wrap(i-1,NX);
              int jp=wrap(j+1,NY), jm=wrap(j-1,NY);
              double sumN = X[idx(ip,j)] + X[idx(im,j)]
                          + X[idx(i,jp)] + X[idx(i,jm)];
              X[id] = (RHS[id] + r*sumN)/(1.0+4.0*r);
            }
          }
        }
        // Black cells
        #pragma omp parallel for collapse(2)
        for(int j=0; j<NY; j++){
          for(int i=0; i<NX; i++){
            if(((i+j)&1)==1){
              int id = idx(i,j);
              int ip=wrap(i+1,NX), im=wrap(i-1,NX);
              int jp=wrap(j+1,NY), jm=wrap(j-1,NY);
              double sumN = X[idx(ip,j)] + X[idx(im,j)]
                          + X[idx(i,jp)] + X[idx(i,jm)];
              X[id] = (RHS[id] + r*sumN)/(1.0+4.0*r);
            }
          }
        }
    }
}

// Compute mean & standard deviation of vector arr
void compute_stats(const std::vector<double> &arr, double &mean, double &stddev) {
    long N = arr.size();
    double sum=0, sum2=0;
    #pragma omp parallel for reduction(+:sum,sum2)
    for(long i=0; i<N; i++){
        sum  += arr[i];
        sum2 += arr[i]*arr[i];
    }
    mean = sum/N;
    stddev = std::sqrt(std::max(0.0, sum2/N - mean*mean));
}

int main(){
    // 0 = sweep alpha; 1 = sweep beta; 2 = sweep DA/DB ratio
    constexpr int SWEEP_PARAM = 0;  

    // --- Define sweep values here ---
    std::vector<double> sweep_vals = { 0.0, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0 };
    // --- Clamp values for the other two parameters ---
    constexpr double alpha_clamp = 0.05;
    constexpr double beta_clamp  = 0.80;
    constexpr double DA_clamp    = 0.05;
    constexpr double DB_clamp    = 1.00;

    // Temporary storage for A, B, and RHS fields
    std::vector<double> A(NX*NY), B(NX*NY), Atemp(NX*NY), Btemp(NX*NY);

    for(double sv : sweep_vals){
        // Set parameters for this run
        double alpha = (SWEEP_PARAM==0 ? sv : alpha_clamp);
        double beta  = (SWEEP_PARAM==1 ? sv : beta_clamp);
        double DA    = (SWEEP_PARAM==2 ? sv*DB_clamp : DA_clamp);
        double DB    = DB_clamp;  // always fixed

        // Homogeneous steady state
        double A0 = alpha + beta;
        double B0 = beta / ((alpha+beta)*(alpha+beta));

        // initialize with noise around steady state
        srand(1234);
        for(int j=0; j<NY; j++){
            for(int i=0; i<NX; i++){
                int id = idx(i,j);
                A[id] = A0 + (((double)rand()/RAND_MAX)-0.5);
                B[id] = B0 + (((double)rand()/RAND_MAX)-0.5);
            }
        }

        // open CSV for this parameter
        char fname[64];
        const char* pname = (SWEEP_PARAM==0?"alpha": SWEEP_PARAM==1?"beta":"diff");
        sprintf(fname,"std_sweep_%s_%.2f.csv", pname, sv);
        std::ofstream out(fname);
        out << "time_step,sigma_A,sigma_B\n";

        // time loop
        for(int step=0; step<=STEPS; step++){
            // explicit reaction
            #pragma omp parallel for collapse(2)
            for(int j=0; j<NY; j++){
              for(int i=0; i<NX; i++){
                int id = idx(i,j);
                double a=A[id], b=B[id];
                double fA = alpha - a + a*a*b;
                double fB = beta  - a*a*b;
                Atemp[id] = a + dt*fA;
                Btemp[id] = b + dt*fB;
              }
            }
            // implicit diffusion
            diffuseGaussSeidel(A, Atemp, DA);
            diffuseGaussSeidel(B, Btemp, DB);

            // output stats every OUTPUT_INTERVAL steps
            if(step % OUTPUT_INTERVAL == 0){
                double meanA, stdA, meanB, stdB;
                compute_stats(A, meanA, stdA);
                compute_stats(B, meanB, stdB);
                out << step*dt << "," << stdA << "," << stdB << "\n";
            }
        }
        out.close();
        std::cout<<"Wrote "<<fname<<"\n";
    }

    return 0;
}
