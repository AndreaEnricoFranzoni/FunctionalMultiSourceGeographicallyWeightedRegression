#ifndef FDAGWR_TRAITS_SUM_HPP
#define FDAGWR_TRAITS_SUM_HPP


#include "include_fdagwr.hpp"
#include "traits_fdagwr.hpp"

#include <iostream>
void
f_sum_test()
{
    Eigen::MatrixXd a(2,2);
    a.setConstant(4);

    Eigen::MatrixXd b(2,2);
    b.setConstant(5);

    Eigen::MatrixXd c = a + b;
std::cout << c <<std::endl;
}



#endif