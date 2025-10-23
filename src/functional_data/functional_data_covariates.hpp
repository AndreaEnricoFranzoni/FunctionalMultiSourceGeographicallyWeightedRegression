// Copyright (c) 2025 Andrea Enrico Franzoni (andreaenrico.franzoni@gmail.com)
//
// This file is part of fdagwr
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of fdagwr and associated documentation files (the fdagwr software), to deal
// fdagwr without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of fdagwr, and to permit persons to whom fdagwr is
// furnished to do so, subject to the following conditions:
//
// fdagwr IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH fdagwr OR THE USE OR OTHER DEALINGS IN
// fdagwr.


#ifndef FDAGWR_FUNCTIONAL_DATA_DATASET_HPP
#define FDAGWR_FUNCTIONAL_DATA_DATASET_HPP


#include "functional_data.hpp"
#include "../basis/basis_include.hpp"
#include "../basis/basis_factory_proxy.hpp"


/*!
* @file functional_data_covariates.hpp
* @brief Contains the class for a dataset of functional data 
* @author Andrea Enrico Franzoni
*/



/*!
* @class functional_data_covariates
* @tparam domain_type the geometry of the basis of the functional data covariates
* @tparam stationarity_t the type of covariates
* @brief class representing a dataset of functional data covariates
*/
template< typename domain_type = FDAGWR_TRAITS::basis_geometry, FDAGWR_COVARIATES_TYPES stationarity_t = FDAGWR_COVARIATES_TYPES::STATIONARY >
    requires fdagwr_concepts::as_interval<domain_type>
class functional_data_covariates
{

private:
    /*!Number of covariates*/
    std::size_t m_q;
    /*!Number of statistical units*/
    std::size_t m_n;
    /*!Functional data covariates*/
    std::vector<functional_data< domain_type,basis_base_class >> m_X;

public:
    /*!
    * @brief Constructor
    * @param coeff vector of Eigen::MatrixXd containing the coefficients of the basis expansion for each covariate
    * @param q the number of covariates
    * @param basis_types vector of strings containing the type of covariates for each covariate
    * @param basis_degrees vector of integers containing the basis degree for each covariate
    * @param basis_numbers vector of integers containing the number of basis for each covariate
    * @param knots Eigen::VectorXd containing the knots over which the basis, for each covariate, are constructed
    * @param factoryBasis factory for creating the basis of each fucntional covariate, their type known at run-time
    */
    functional_data_covariates(const std::vector<FDAGWR_TRAITS::Dense_Matrix> & coeff,
                               std::size_t q,
                               const std::vector<std::string> & basis_types,
                               const std::vector<std::size_t> & basis_degrees,
                               const std::vector<std::size_t> & basis_numbers,
                               const FDAGWR_TRAITS::Dense_Vector & knots,
                               const basis_factory::basisFactory& factoryBasis)
                        :
                            m_q(q)
                        {
                            //input coherency
                            m_n = m_q > 0 ? coeff[0].cols() : 0;

                            //creating the vector of fd
                            m_X.reserve(m_q);
                            for(std::size_t i = 0; i < m_q; ++i)
                            {
                                std::unique_ptr<basis_base_class<domain_type>> basis_i = factoryBasis.create(basis_types[i],knots,basis_degrees[i],basis_numbers[i]);
                                functional_data< domain_type,basis_base_class > x_i(std::move(coeff[i]),std::move(basis_i));
                                m_X.push_back(x_i);
                            }
                        }
    
    /*!
    * @brief Getter for the number of covariates
    */
    std::size_t q() const {return m_q;}

    /*!
    * @brief Getter for the number of statistical units
    */
    std::size_t n() const {return m_n;}

    /*!
    * @brief Function for evaluating one unit of one covariate in a given location
    * @param loc the abscissa over which evaluating
    * @param cov_i which covariate evaluating
    * @param unit_j which unit evaluating
    * @return a double with the evaluation of a unit, specific covariate
    */
    double
    eval(const double &loc, std::size_t cov_i, std::size_t unit_j)
    const
    {
        return m_X[cov_i].eval(loc,unit_j); //evaluation of unit_j-th of covariate cov_i-th in location loc (starting from 0)
    }
};

#endif  /*FDAGWR_FUNCTIONAL_DATA_DATASET_HPP*/