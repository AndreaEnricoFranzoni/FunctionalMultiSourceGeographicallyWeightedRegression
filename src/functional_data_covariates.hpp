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
// OUT OF OR IN CONNECTION WITH PPCKO OR THE USE OR OTHER DEALINGS IN
// fdagwr.


#ifndef FDAGWR_FUNCTIONAL_DATA_DATASET_HPP
#define FDAGWR_FUNCTIONAL_DATA_DATASET_HPP


#include "functional_data.hpp"
#include "basis_include.hpp"
#include "basis_factory_proxy.hpp"


template< typename domain_type = FDAGWR_TRAITS::basis_geometry, FDAGWR_COVARIATES_TYPES stationarity_t = FDAGWR_COVARIATES_TYPES::STATIONARY >
    requires fdagwr_concepts::as_interval<domain_type>
class functional_data_covariates
{
private:
    /*!How many covariates*/
    std::size_t m_q;
    /*!How many statistical units*/
    std::size_t m_n;
    /*!Functional data covariates*/
    std::vector<functional_data< domain_type,basis_base_class >> m_X;

public:
    /*!Constructor*/
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
                            m_n = m_q > 0 ? coeff[0].cols() : 0;

                            m_X.reserve(m_q);
                            for(std::size_t i = 0; i < m_q; ++i)
                            {
                                std::unique_ptr<basis_base_class<domain_type>> basis_i = factoryBasis.create(basis_types[i],knots,basis_degrees[i],basis_numbers[i]);
                                m_X.emplace_back(std::move(coeff[i]),std::move(basis_i));
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

    /*!Evaluation function*/
    double
    eval(double loc, std::size_t cov_i, std::size_t unit_j)
    const
    {
        return m_X[cov_i].eval(loc,unit_j); //evaluation of unit_j-th of covariate cov_i-th in location loc (starting from 0)
    }
};

#endif  /*FDAGWR_FUNCTIONAL_DATA_DATASET_HPP*/