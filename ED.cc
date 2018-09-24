#include <iostream>
#include <vector>
#include <complex>
#include <math.h>
#include <sys/time.h>

#include "mkl.h"
#include "mkl_lapacke.h"

typedef long long int LLInt;

double seconds()
{
  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *)0 );
  sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
  return sec;
}

LLInt basis_size(LLInt l, 
                 LLInt n) 
{
  double size = 1.0;
  for(LLInt i = 1; i <= (l - n); ++i){
    size *= (static_cast<double> (i + n) / static_cast<double> (i));  
  }

  return floor(size + 0.5);
}

LLInt first_int(LLInt n)
{
  LLInt first = 0;
  for(LLInt i = 0; i < n; ++i){
    first += 1 << i;
  }

  return first;
}

void construct_int_basis(LLInt *int_basis, 
                         LLInt l, 
                         LLInt n)
{
  LLInt w;
  LLInt first = first_int(n);
  LLInt basis_s = basis_size(l, n);  

  int_basis[0] = first;

  for(LLInt i = 1; i < basis_s; ++i){
    LLInt t = (first | (first - 1)) + 1;
    w = t | ((((t & -t) / (first & -first)) >> 1) - 1);
    
    int_basis[i] = w;

    first = w;
  }
}

LLInt binsearch(const LLInt *array, LLInt len, LLInt value)
{
  if(len == 0) return -1;
  LLInt mid = len / 2;

  if(array[mid] == value) 
    return mid;
  else if(array[mid] < value){
    LLInt result = binsearch(array + mid + 1, len - (mid + 1), value);
    if(result == -1) 
      return -1;
    else
      return result + mid + 1;
  }
  else
    return binsearch(array, mid, value);
}

void construct_XXZ_matrix(double *ham_mat,
                          double *curr_mat,
                          double *kin_mat,
                          LLInt *int_basis,
                          std::vector<double> &alpha,
                          std::vector<double> &delta,
                          std::vector<double> &h,
                          LLInt l,
                          LLInt n,
                          bool periodic_boundaries = false)
{
  LLInt basis_s = basis_size(l, n);
  for(LLInt state = 0; state < basis_s; ++state){

    LLInt bs = int_basis[state];
    double vi = 0.0;
    double mag_term = 0.0;

    for(LLInt site = 0; site < l; ++site){
      LLInt bitset = bs;
      if(bitset & (1 << site))
        mag_term += h[site];
      else
        mag_term -= h[site];
      
      if(!periodic_boundaries){
        if(site == l - 1)
          continue;
      }

      if(bitset & (1 << site)){
        if(bitset & (1 << ( (site + 1) % l ) ) ){
          vi += delta[site];
          continue;
        }
        else{
          vi -= delta[site];
          bitset ^= 1 << site;
          bitset ^= 1 << (site + 1) % l;
          LLInt match_ind1 = binsearch(int_basis, basis_s, bitset);
          ham_mat[ (state * basis_s) + match_ind1 ] = 2.0 * alpha[site],0;
          curr_mat[ (state * basis_s) + match_ind1 ] = -2.0 * alpha[site];
          kin_mat[ (state * basis_s) + match_ind1 ] = -2.0 * alpha[site];
          continue;
        }     
      } // End spin up case
      else{
        if(bitset & (1 << ( (site + 1) % l ) ) ){
          vi -= delta[site];
          bitset ^= 1 << site;
          bitset ^= 1 << (site + 1) % l;
          LLInt match_ind2 = binsearch(int_basis, basis_s, bitset);
          ham_mat[ (state * basis_s) + match_ind2 ] = 2.0 * alpha[site],0;
          curr_mat[ (state * basis_s) + match_ind2 ] = 2.0 * alpha[site];
          kin_mat[ (state * basis_s) + match_ind2 ] = -2.0 * alpha[site];
          continue;
        }
        else{
          vi += delta[site];
          continue;
        }
      } // End spin down case
    } // End site loop
  ham_mat[ (state * basis_s) + state ] = vi + mag_term;
  } // End state loop
}

void rotate(double *a, double *rot, LLInt basis_s)
{
  double *buffer = new double[basis_s * basis_s]();
  cblas_dgemm(CblasRowMajor,
              CblasNoTrans,
              CblasNoTrans,
              basis_s,
              basis_s,
              basis_s,
              1.0,
              a,
              basis_s,
              rot,
              basis_s,
              0.0,
              buffer,
              basis_s);
  cblas_dgemm(CblasRowMajor,
              CblasTrans,
              CblasNoTrans,
              basis_s,
              basis_s,
              basis_s,
              1.0,
              rot,
              basis_s,
              buffer,
              basis_s,
              0.0,
              a,
              basis_s);

  delete [] buffer;
}

void print_matrix(double *mat, 
                  LLInt n)
{
  for(LLInt i = 0; i < n; ++i){
    for(LLInt j = 0; j < n; ++j){
      std::cout << mat[ (i * n) + j ] << " ";
    }
  std::cout << std::endl;
  }
}

int main(int argc, char **argv)
{
  LLInt l = 777;
  LLInt n = 777;
  double alpha_val = 0.777;
  double delta_val = 0.777;
  double h_val = 0.777;
  bool periodic = true;
  int drude_bool = 0;
  int freq_bool = 0;

  if(argc != 17){
    std::cerr << "Usage: " << argv[0] << 
      " --l [sites] --n [fill] --alpha [alpha] --delta [delta] --h [h] --periodic [bool] --drude [0/1] --freq [0/1]" 
        << std::endl;
    exit(1);
  }

  for(int i = 0; i < argc; ++i){
    std::string str = argv[i];
    if(str == "--l") l = atoi(argv[i + 1]);
    else if(str == "--n") n = atoi(argv[i + 1]);
    else if(str == "--alpha") alpha_val = atof(argv[i + 1]);
    else if(str == "--delta") delta_val = atof(argv[i + 1]);
    else if(str == "--h") h_val = atof(argv[i + 1]);
    else if(str == "--periodic") periodic = atoi(argv[i + 1]);
    else if(str == "--drude") drude_bool = atoi(argv[i + 1]);
    else if(str == "--freq") freq_bool = atoi(argv[i + 1]);
    else continue;
  }

  if(l == 777 || n == 777 || alpha_val == 0.777 || delta_val == 0.777 || h_val == 0.777){
    std::cerr << "Error setting parameters" << std::endl;
    std::cerr << "Usage: " << argv[0] << 
      " --l [sites] --n [fill] --alpha [alpha] --delta [delta] --h [h] --periodic [bool] --drude [0/1] --freq [0/1]" 
        << std::endl;
    exit(1);
  }

  std::vector<double> alpha(l, alpha_val);
  std::vector<double> delta(l, delta_val);
  std::vector<double> h(l, 0.0);
  // Impurity model
  h[l / 2] = h_val;
  // Staggered field
  //for(LLInt i = 0; i < l; i += 2){
  //  h[i] = -1.0 * h_val;
  //}

  LLInt basis_s = basis_size(l, n);

  std::cout << std::fixed;
  std::cout.precision(1);
  std::cout << "# Parameters:" << std::endl;
  std::cout << "# L = " << l << std::endl;
  std::cout << "# N = " << n << std::endl;
  std::cout << "# Alpha = " << "[";
  for(LLInt i = 0; i < (l - 1); ++i){
    std::cout << alpha[i] << ", ";
  }
  std::cout << alpha[l - 1] << "]" << std::endl;
  std::cout << "# Delta = " << "[";
  for(LLInt i = 0; i < (l - 1); ++i){
    std::cout << delta[i] << ", ";
  }
  std::cout << delta[l - 1] << "]" << std::endl;
  std::cout << "# h = " << "[";
  for(LLInt i = 0; i < (l - 1); ++i){
    std::cout << h[i] << ", ";
  }
  std::cout << h[l - 1] << "]" << std::endl;

  std::cout << std::fixed;
  std::cout.precision(7);
  // Basis
  double tic_i = seconds();
  double tic = seconds();
  LLInt *int_basis = new LLInt[basis_s];
  construct_int_basis(int_basis, l, n);

  // Hamiltonian matrix, current and kinetic energy
  double *ham_mat = new double[basis_s * basis_s]();
  double *curr_mat = new double[basis_s * basis_s]();
  double *kin_mat = new double[basis_s * basis_s]();

  construct_XXZ_matrix(ham_mat,
                       curr_mat,
                       kin_mat,
                       int_basis,
                       alpha,
                       delta,
                       h,
                       l,
                       n,
                       periodic);
  double toc = seconds();
  std::cout << "# Time constructing matrices: " 
    << (toc - tic) << std::endl;
  
  delete [] int_basis;
  
  // Diag
  double *eigvals = new double[basis_s];
  MKL_INT info;

  //std::cout << "# Calculating eigen..." << std::endl;
  tic = seconds();
  info = LAPACKE_dsyevd( LAPACK_ROW_MAJOR, 'V', 'U', basis_s, ham_mat, basis_s, eigvals );
  if(info > 0){
    std::cout << "Eigenproblem" << std::endl;
    exit(1);
  }
  toc = seconds();
  std::cout << "# Time eigen: " 
    << (toc - tic) << std::endl;

  // Rotate J and T
  std::cout << "# Rotating matrices..." << std::endl;
  rotate(curr_mat, ham_mat, basis_s);
  rotate(kin_mat, ham_mat, basis_s);

  // Calculating probs
  std::cout << "# Calculating probs..." << std::endl;
  double beta = 0.001;
  double *probs = new double[basis_s];
  for(LLInt ind = 0; ind < basis_s; ++ind){
    probs[ind] = std::exp(-1.0 * beta * eigvals[ind]);
  }
  double z_sum = cblas_dasum(basis_s, probs, 1);
  z_sum = 1.0 / z_sum;
  cblas_dscal(basis_s, z_sum, probs, 1);

  // Compute expectation value of T
  std::cout << "# Calculating <-T>..." << std::endl;
  double t_exp = 0.0;
  for(LLInt ind = 0; ind < basis_s; ++ind){
    t_exp += kin_mat[ (ind * basis_s) + ind ] * probs[ind];
  }
  delete [] kin_mat;

  // Compute Drude weight
  if(drude_bool){
    std::cout << "# Calculating D_N..." << std::endl;
    double drude = 0.0;
    for(LLInt n = 0; n < basis_s; ++n){
      for(LLInt m = (n + 1); m < basis_s; ++m){
        if( fabs(eigvals[n] - eigvals[m]) > 1.0e-8 ){
          double jnm = curr_mat[(n * basis_s) + m];
          drude += 2.0f * ( (probs[n] - probs[m]) / (eigvals[m] - eigvals[n]) * ( jnm * jnm ) );
        }
      }
    }
    double val = ( t_exp - drude ) / l;

    double toc_f = seconds();

    std::cout << "# Done..." << std::endl;
    std::cout << "# Basis size = " << basis_s << std::endl;
    std::cout << "# <-T> = " << t_exp << std::endl;
    std::cout << "# Sum term = " << drude << std::endl;
    std::cout << "# D_N / (<-T>/N) = " << (val / (t_exp / l)) << std::endl;
    std::cout << "# Total time = " << toc_f - tic_i << std::endl;
  }

  if(freq_bool){
    std::cout << "# Basis size = " << basis_s << std::endl;

    double pi = 3.1415927;
    double bin_size = 0.02;
    double max_omega = 4.0;

    int n_bins = static_cast<int>( std::ceil(max_omega / bin_size) );
    double *sigmas = new double[n_bins]();

    for(LLInt n = 0; n < basis_s; ++n){
      for(LLInt m = 0; m < (n + 1); ++m){
        double omega = (eigvals[n] - eigvals[m]);
        if( fabs(omega) < 1.0e-8 )
          continue;
        if(omega < 0.0 || omega >= max_omega)
          continue;
        double jnm = curr_mat[(n * basis_s) + m];
        int bin_index = static_cast<int>( std::floor(omega / bin_size) );
        sigmas[bin_index] += ( (pi / l) * ((1.0 - std::exp(-1.0 * beta * omega)) / omega) 
          * probs[n] * ( jnm * jnm ) );
      }
    }

    double area = 0.0;
    for(LLInt i = 0; i < n_bins; ++i)
      area += sigmas[i] * bin_size;

    std::cout << "# Area = " << area / bin_size << std::endl;
    std::cout << "# pi*<-T>/2L = " << t_exp * pi / (2.0 * l) << std::endl;

    for(LLInt i = 0; i < n_bins; ++i)
      sigmas[i] = sigmas[i] * (1.0 / (bin_size * pi * (t_exp / l)));

    std::cout << "# Normalising..." << std::endl;
    area = 0.0;
    for(LLInt i = 0; i < n_bins; ++i)
      area += sigmas[i] * bin_size;

    std::cout << "# New Area = " << area << std::endl;

    std::cout << "# w Re[sigma(w)/pi*(<-T>/L)" << std::endl;
    std::cout << "0.0000000 0.0000000" << std::endl;
    for(LLInt i = 0; i < n_bins; ++i)
      std::cout << (i + 1) * bin_size << " " << sigmas[i] << std::endl;

    double toc_f = seconds();
    std::cout << "# Total time = " << toc_f - tic_i << std::endl;

    delete [] sigmas;
  }

  delete [] curr_mat;
  delete [] ham_mat;
  delete [] eigvals;
  delete [] probs;

  return 0;
}
