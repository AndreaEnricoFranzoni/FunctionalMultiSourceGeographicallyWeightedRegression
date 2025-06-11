# Functional Multi-Source Geographically Weighted Regression

**`fdagwr`**  is a C++ based R package for performing geographically weighted regression in the case of a functional response and functional covariates. The model has been proposed in a scalar setting by [Caramenti et Al.](#ref-caramenti). [Bortolotti et Al.](#ref-bortolotti) extended the work for the functional case if having only stationary coefficients (i.e. not varying with respect to the geographical position). [Fervari, Menafoglio and Bortolotti](#ref-fervari) propose a union of the previous twos, in the form of:

$y_{i}(t) = \sum _{j \in C} x_{C,ij}(t) \beta_{C,j}(t) + \sum _{j \in E} x_{E,ij}(t) \beta_{E,j}(p_{E,i},t) + \sum _{j \in S} x_{S,ij}(t) \beta_{S,j}(p_{S,i},t) + \epsilon_{i}(t), \;\; t \in T, i \in 1,\dots,n $

The geographical domains of events (where the event occurs) and stations (where the events is measured) are $D_E$ and $D_S$, and may not necessarily coincide. Consequently,
the specific spatial location of the i-th event pE,i = (uE,i,vE,i) ∈DE and i-th station pS,i = (uS,i,vS,i) ∈DS.




# Prerequisites

R has to be updated at least to 4.0.0 version. If Windows is used, R version has to be at least 4.4.0.

On R console:
~~~
library(devtools)
~~~

Or, alternatively, if not installed:
~~~
install.packages("devtools")
library(devtools)
~~~

**`fdagwr`** depends also on having Fortran, Lapack, BLAS and, if parallel version is used, OpenMP installed.
Depending on the operative system, the instructions to set up everything can be found [here below](#prerequisites-depending-on-operative-system).




# Installation

To install the package:

- Parallel version
~~~
devtools::install_github("AndreaEnricoFranzoni/FunctionalMultiSourceGeographicallyWeightedRegression")
~~~


And upload the library in the R environment
~~~
library(fdagwr)
~~~



Due to the high number of warnings, to disable them can be useful adding as argument of `install_github`
~~~
quiet=TRUE
~~~


If problem related to dependencies arises when installing, also the argument 
~~~
dependencies = TRUE
~~~
could be useful



# Prerequisites: depending on operative system

More detailed documentation can be found in [this section](https://cran.r-project.org) of `The R Manuals`.
Although installing **`fdagwr`** should automatically install all the R dependecies, could be worth trying to install them manaully if an error occurs.
~~~
install.packages("Rcpp")
install.packages("RcppEigen")
library(Rcpp)
library(RcppEigen)
~~~

## macOS

1. **Fortran**:  unlike Linux and Windows, Fortran has to be installed on macOS: instructions in this [link](https://cran.r-project.org/bin/macosx/tools/). Lapack and BLAS will be consequently installed.

2. **OpenMP**: unlike Linux and Windows, OMP is not installed by default on macOS. Open the terminal and digit the following commands.

- **Homebrew**
  - Check the presence of Homebrew
    ~~~
    brew --version
    ~~~
    - If this command does not give back the version of Homebrew, install it according to the macOS version
    ~~~
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ~~~
      1. *M1* or *M2*
      ~~~
      echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
      eval "$(/opt/homebrew/bin/brew shellenv)"
      ~~~
      2. *Intel*
      ~~~
      echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
      eval "$(/usr/local/bin/brew shellenv)"
      ~~~

- **OMP**
  - Once Homebrew is set, check the presence of OMP
    ~~~
    brew list libomp
    ~~~
  - Install it in case of negative output
    ~~~
    brew install libomp
    ~~~

- **LLVM**
   
  [LLVM](https://llvm.org) is needed to configure clang on macOS in order to use an external OMP version
  - Check its presence
  ~~~
  llvm-config --version
  ~~~
  - Eventually, download it
  ~~~
  brew install llvm
  ~~~

- Check that the command that opens the graphic window used by the visualization results functions
  ~~~
  quartz()
  ~~~
  works on R, by digiting it on R console. If not, look [here](https://www.xquartz.org).

## Windows

- **Rtools**: can be installed from [here](https://cran.r-project.org/bin/windows/Rtools/). Version 4.4 is needed to install parallel version.

- Check that the command that opens the graphic window used by the visualization results functions
  ~~~
  windows()
  ~~~
  works on R, by digiting it on R console.

## Linux

- Linux installation depends on its distributor. All the commands here reported will refer to Ubuntu and Debian distributors. Standard developement packages have to be installed.   In Ubuntu and Debian, for example, all the packages have been collected into a single one, that is possible to install digiting into the terminal:

   ~~~
  sudo apt install r-base-dev
  sudo apt install build-essential
   ~~~

- Check that the command that opens the graphic window used by the visualization results functions
  ~~~
  x11()
  ~~~
  works on R, by digiting it on R console.

## Linux image
Before being able to run the commands explained above for Linux, R has to be downloaded. Afterward, the installation of Fortran, Lapack, BLAS, devtools and its dependecies can be done by digiting into the terminal:
   ~~~
sudo apt-get update
sudo apt install gfortran
sudo apt install liblapack-dev libblas-dev
   ~~~
   ~~~
sudo apt-get install libcurl4-openssl-dev
sudo apt-get install libssl-dev
sudo apt-get install libz-dev
sudo apt-get install -y libcurl4-openssl-dev libssl-dev libxml2-dev
sudo apt install zlib1g-dev
sudo apt install -y libfreetype6-dev libfontconfig1-dev libharfbuzz-dev libcairo2-dev libpango1.0-dev pandoc
   ~~~



# Bibliography 
1. <a id="ref-fervari"></a> **Fervari M., Menafoglio A., Bortolotti T.**, `A Functional Multi-Source Geographically Weighted Regression for
Ground Motion Modelling in Italy`, 2025

2. <a id="ref-bortolotti"></a> **Bortolotti T., Peli R., Lanzano G., Sgobba S., Menafoglio A.**, `Weighted
functional data analysis for the calibration of a ground motion model in italy`, *RJournal of the American
Statistical Association*, 119(547):1697–1708, 2024, https://doi.org/10.1080/01621459.2023.2300506

3. <a id="ref-caramenti"></a> **Caramenti L., Menafoglio A., Sgobba S., Lanzano G.**, `Multi-source geographically
weighted regression for regionalized ground-motion models`, *Spatial Statistics*, 47:100610, 2022, https://doi.org/10.1016/j.spasta.2022.100610

4. <a id="ref-Rcpp"></a> **Eddelbuettel D., Francois R., Allaire J., Ushey K., Kou Q., Russell N., Ucar I., Bates D., Chambers J.**, `Rcpp: Seamless R and C++ Integration`, *R package version 1.0.13-1*, 2024, https://cran.r-project.org/web/packages/Rcpp/citation.html

5. <a id="ref-eigen"></a> **Guennebaud G., Jacob B., et Al.**, `Eigen v3.4`, 2021, https://eigen.tuxfamily.org/index.php?title=Main_Page

6. <a id="ref-pacsexamples"></a> **Formaggia L., et Al.**, `EXAMPLES AND EXERCISES FOR AMSC and APSC (PACS) COURSES`, 2024, https://github.com/pacs-course/pacs-examples
