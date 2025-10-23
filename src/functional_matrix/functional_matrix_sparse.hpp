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


#ifndef FUNCTIONAL_MATRIX_SPARSE_HPP
#define FUNCTIONAL_MATRIX_SPARSE_HPP


#include "functional_matrix_expression_wrapper.hpp"
#include "functional_matrix_utils.hpp"
#include <utility>
#include <vector>
#include <cassert>

#include <iostream>


/*!
* @file functional_matrix_sparse.hpp
* @brief Contains the definition of a sparse matrix containing univariate 1D domain std::function objects
* @author Andrea Enrico Franzoni
*/




/*!
* @class functional_matrix_sparse
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @brief Class for sparse matrices storing, column-wise, in compress format (CSC), univariate 1D domain std::function objects
* @details Static polymorphism: deriving from a expression for expression templates
* @note Functions are stored column-wise, compress format (CSC)
* @todo PROBLEM IN CONVERTING FROM DENSE TO SPARSE
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class functional_matrix_sparse : public Expr< functional_matrix_sparse<INPUT,OUTPUT>, INPUT, OUTPUT >
{
    /*!std::function object stored*/
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    /*!std::function input type*/
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;

    /*!null function (const version)*/
    inline static const F_OBJ m_null_function = [](F_OBJ_INPUT x){ return static_cast<OUTPUT>(0);};
    /*!null function (non-const verion)*/
    inline static F_OBJ m_null_function_non_const = [](F_OBJ_INPUT x){ return static_cast<OUTPUT>(0);};

private:
    /*!Number of rows.*/
    std::size_t m_rows;
    /*!Number of cols.*/                              
    std::size_t m_cols;
    /*!Number of non-zero elements*/                              
    std::size_t m_nnz;
    /*!Vector that contains row indices. Size is nnz, and contains, for each value of m_data, their row. NB: for each col, idxs have to be ordered*/           
    std::vector<std::size_t> m_rows_idx;        
    /*! @brief Vector that contains col pointer. Size is m_cols+1, elem i-th of the vector contains the number of nnz elements up to col i+1-th (not inlcuding it) 
    * (list of m_data indexes where each column starts).
    * @example element 0-th of m_cols_idx indicates how many elements are present, in total before the first column. Element 1st indicates how many elements are present, in total before the second column
    */  
    std::vector<std::size_t> m_cols_idx;
    /*!Container for the std::function. The storage order is column-wise (CSC)*/
    std::vector< F_OBJ > m_data;    

public:

    /*!
    * @brief Default constructor
    */
    functional_matrix_sparse() = default;

    /*!
    * @brief Constructor
    * @param fm vector of std::function objects
    * @param n_rows number of rows
    * @param n_cols number of columns
    * @param rows_idx vector of integers containing, for each element in fm, in which row it is (size if size of fm)
    * @param cols_idx vector of integers containing, for each column, the cumulative number of elements up to that column (size if n_cols + 1)
    */
    functional_matrix_sparse(std::vector< F_OBJ > const &fm,
                             std::size_t n_rows,
                             std::size_t n_cols,
                             std::vector<std::size_t> const &rows_idx,
                             std::vector<std::size_t> const &cols_idx)
                :   m_rows(n_rows), m_cols(n_cols), m_nnz(fm.size()), m_rows_idx{rows_idx}, m_cols_idx{cols_idx}, m_data{fm} 
                {
                    //cheack input consistency
                    assert((void("Number of rows times number of cols has to be greater than the number of stored functions"), m_rows * m_cols > m_data.size()));
                    assert((void("Number of rows indeces has to be equal to the number of stored elements"), m_rows_idx.size() == m_data.size()));
                    assert((void("Number of cols indeces has to be equal to the number of cols + 1"), m_cols_idx.size() == (m_cols + 1)));
                    assert(m_cols_idx.front() == 0);
                    assert(m_cols_idx.back() == m_nnz);
                }

    /*!
    * @brief Constructor with move semantic
    * @param fm vector of std::function objects
    * @param n_rows number of rows
    * @param n_cols number of columns
    * @param rows_idx vector of integers containing, for each element in fm, in which row it is (size if size of fm)
    * @param cols_idx vector of integers containing, for each column, the cumulative number of elements up to that column (size if n_cols + 1)
    */
    functional_matrix_sparse(std::vector< F_OBJ > &&fm,
                             std::size_t n_rows,
                             std::size_t n_cols,
                             std::vector<std::size_t> &&rows_idx,
                             std::vector<std::size_t> &&cols_idx)
                :   m_rows(n_rows), m_cols(n_cols), m_nnz(fm.size()), m_rows_idx{std::move(rows_idx)}, m_cols_idx{std::move(cols_idx)}, m_data{std::move(fm)} 
                {
                    //cheack input consistency
                    assert((void("Number of rows times number of cols has to be greater than the number of stored functions"), m_rows * m_cols > m_data.size()));
                    assert((void("Number of rows indeces has to be equal to the number of stored elements"), m_rows_idx.size() == m_data.size()));
                    assert((void("Number of cols indeces has to be equal to the number of stored elements + 1"), m_cols_idx.size() == (m_cols + 1)));
                    assert(m_cols_idx.front() == 0);
                    assert(m_cols_idx.back() == m_nnz);
                }

    /*!
    * @brief Copy constructor
    */
    functional_matrix_sparse(functional_matrix_sparse const &) = default;

    /*!
    * @brief Move constructor
    */    
    functional_matrix_sparse(functional_matrix_sparse &&) = default;

    /*!
    * @brief Copy assignment
    */    
    functional_matrix_sparse &operator=(functional_matrix_sparse const &) = default;

    /*!
    * @brief Move assignment
    */    
    functional_matrix_sparse &operator=(functional_matrix_sparse &&) = default;


    /*!
    * @brief Constructor that builds a functional_matrix_sparse from an Expr 
    * @tparam T template param indicating the derived type from expression from which the cast is done
    * @param e expression from which constructing a functional_matrix_sparse
    * @details necessary for ETs design
    * @todo IT LACKS A WAY TO CHOOSE IF INSERTING A FUNCTION ONLY IF IT IS NON_null_f: RETURNS A SPARSE WITH THE 0s SAVES
    */
    template <class T> 
    functional_matrix_sparse(const Expr<T,INPUT,OUTPUT> &e)
        :   m_data()
    {
        //casting
        const T &et(e); 
        m_rows = et.rows(); 
        m_cols = et.cols();
        m_nnz = et.size(); 

        //reserving the correct amount of capacity
        m_data.reserve(et.size());
        m_rows_idx.reserve(et.size());
        m_cols_idx.reserve(et.cols()+1);
        m_cols_idx.emplace_back(static_cast<std::size_t>(0));           //the first element of the cumulative number of elements per columns is always 0
        //counter for the cumulative number of elements along columns
        std::size_t counter_cols_elem = 0;

        //looping as this for column-wise storage
        for(std::size_t j = 0; j < et.cols(); ++j){
            for(std::size_t i = 0; i < et.rows(); ++i){
                //inserting only if non-null: IT LACKS THIS IF CONDITION (NOW IT COMPARE THE ADDRESS OF THE NON CONST NULL FUNCTION)
                m_data.emplace_back(et(i,j));
                if(&m_data.back() == &functional_matrix_sparse<INPUT,OUTPUT>::m_null_function_non_const)
                {
                    m_data.pop_back();
                }
                else
                {
                    m_rows_idx.emplace_back(i);
                    counter_cols_elem += 1;
                }
            }
            m_cols_idx.emplace_back(counter_cols_elem);
        }
    }

    /*!
    * @brief Copy assignment from an Expr
    * @tparam T template param indicating the derived type from expression from which the cast is done
    * @param e expression from which constructing a functional_matrix_sparse
    * @details necessary for ETs design
    * @todo IT LACKS A WAY TO CHOOSE IF INSERTING A FUNCTION ONLY IF IT IS NON_null_f: RETURNS A SPARSE WITH THE 0s SAVES
    */
    template <class T>
    functional_matrix_sparse &
    operator=(const Expr<T,INPUT,OUTPUT> &e)
    {
        //casting
        const T &et(e); 
        m_rows = et.rows(); 
        m_cols = et.cols();
        m_nnz = et.size();

        //reserving the correct amount of capacity
        m_data.reserve(et.size());
        m_rows_idx.reserve(et.size());
        m_cols_idx.reserve(et.cols()+1);
        m_cols_idx.emplace_back(static_cast<std::size_t>(0));           //the first element of the cumulative number of elements per columns is always 0
        //counter for the cumulative number of elements along columns
        std::size_t counter_cols_elem = 0;
        

        for(std::size_t j = 0; j < et.cols(); ++j)
        {
            for(std::size_t i = 0; i < et.rows(); ++i)
            {
                m_data.emplace_back(et(i,j));
                //inserting only if non-null: IT LACKS THIS IF CONDITION (NOW IT COMPARE THE ADDRESS OF THE NON CONST NULL FUNCTION)
                if(&m_data.back() == &functional_matrix_sparse<INPUT,OUTPUT>::m_null_function_non_const)    
                {
                    m_data.pop_back();
                }
                else
                {
                    m_rows_idx.emplace_back(i);
                    counter_cols_elem += 1;
                }
            }
            m_cols_idx.emplace_back(counter_cols_elem);
        }

        return *this;
    }

    /*!
    * @brief Checking presence of row idx-th: at least a nnz element in that row.
    * @param idx index of the row whom presence is checked
    * @return true if at least a non-zero element is in that row, false otherwise
    */
    bool
    check_row_presence(std::size_t idx) 
    const
    {
        //checking that the passed index is coherent with the matrix dimension
        assert(idx < m_rows);            
        //it is sufficient that in the vector containing the rows idx there is once idx: need to go with std::find instead of std::binary_search since elements are not ordered
        return std::find(m_rows_idx.cbegin(),m_rows_idx.cend(),idx) != m_rows_idx.cend();
    }

    /*!
    * @brief Checking presence of column idx-th: at least a nnz element in that column.
    * @param idx index of the column whom presence is checked
    * @return true if at least a non-zero element is in that column, false otherwise
    */
    bool
    check_col_presence(std::size_t idx) 
    const
    {
        //checking that the passed index is coherent with the matrix dimension
        assert(idx < m_cols); 
        //it is sufficient that between col i-th and i+1-th there is an increment in number of elements
        return m_cols_idx[idx] < m_cols_idx[idx+1];           
    }

    /*!
    * @brief Checking presence of element (i,j)
    * @param i index of the row of the element whom presence is checked
    * @param j index of the column of the element whom presence is checked
    * @return true if the element is present, false otherwise
    */
    bool
    check_elem_presence(std::size_t i, std::size_t j) 
    const
    {
        //checking if there row i and col j are present
        if(!this->check_col_presence(j) || !this->check_row_presence(i)){   return false;}
        //searching if within the row indeces of the col there is the one requested. Since, for each column, elements of m_rows_idx are ordered, it is possible to relay on std::binary_search
        return std::binary_search(std::next(m_rows_idx.cbegin(),m_cols_idx[j]),         //start of rows idx in col j
                                  std::next(m_rows_idx.cbegin(),m_cols_idx[j+1]),       //end of rows idx in col j
                                  i);                                                   //row index that has to be in the row
    }

    /*!
    * @brief Returns a reference to the element (i,j), non-const reference
    * @param i index of the element row
    * @param j index of the element column
    * @return a reference to the element (i,j)
    * @note being the sparse matrix compressed, it is possible to modify only element already there
    */
    F_OBJ &
    operator()
    (std::size_t i, std::size_t j)
    {
        //checking input consistency
        assert(i < m_rows && j < m_cols); 
        //check the col presence
        if(!this->check_col_presence(j)) {   return this->m_null_function_non_const;}
        //looking at the position of row of the element in the range indicated by the right column. Since, for each column, elements of m_rows_idx are ordered, it is possible to relay on std::lower_bound (finds the first element, in an ordered range, >= value)
        auto elem = std::lower_bound(std::next(m_rows_idx.cbegin(),m_cols_idx[j]),
                                     std::next(m_rows_idx.cbegin(),m_cols_idx[j+1]),
                                     i);
        //taking the distance from the begin to retrain the position in the value's vector, if present, null function if not (since finds the first element, in an ordered range, >= value, it is necessary to check it also)
        return (elem!=std::next(m_rows_idx.cbegin(),m_cols_idx[j+1]) && *elem==i) ? m_data[std::distance(m_rows_idx.cbegin(),elem)] : this->m_null_function_non_const; 
    }

    /*!
    * @brief Returns a reference to the element (i,j), const reference
    * @param i index of the element row
    * @param j index of the element column
    * @return a const reference to the element (i,j)
    */
    F_OBJ
    operator()
    (std::size_t i, std::size_t j)
    const
    {
        //checking input consistency
        assert(i < m_rows && j < m_cols); 
        //check the col presence
        if(!this->check_col_presence(j)) {   return this->m_null_function;}
        //looking at the position of row of the element in the range indicated by the right column. Since, for each column, elements of m_rows_idx are ordered, it is possible to relay on std::lower_bound
        auto elem = std::lower_bound(std::next(m_rows_idx.cbegin(),m_cols_idx[j]),
                                     std::next(m_rows_idx.cbegin(),m_cols_idx[j+1]),
                                     i);
        //taking the distance from the begin to retrain the position in the value's vector, if present, null function if not
        return (elem!=std::next(m_rows_idx.cbegin(),m_cols_idx[j+1]) && *elem==i) ? m_data[std::distance(m_rows_idx.cbegin(),elem)] : this->m_null_function; 
    }

    /*!
    * @brief Rows size
    * @return the number of rows
    */
    std::size_t
    rows() 
    const
    {
        return m_rows;
    }

    /*!
    * @brief Cols size
    * @return the number of columns
    */
    std::size_t
    cols() 
    const
    {
        return m_cols;
    }

    /*!
    * @brief Number of elements stored
    * @return the number of non-zero elements, elements actually stored
    */
    std::size_t
    size() 
    const
    {
        return m_nnz;
    }

    /*!
    * @brief Row indices of the non-zero elements
    * @return a std::vector with the row indeces of the stored elements
    */
    std::vector<std::size_t>
    rows_idx()
    const
    {
        return m_rows_idx;
    }

    /*!
    * @brief Cumulative number of elements up to column i-th
    * @return a std::vector with the cumulative number of elements stored up to column i-th
    */
    std::vector<std::size_t>
    cols_idx()
    const
    {
        return m_cols_idx;
    }

    /*!
    * @brief Transposing the functional sparse matrix
    */
    void
    transposing()
    {
        //row vector ==> col vector
        if(m_rows == static_cast<std::size_t>(1) && m_cols != static_cast<std::size_t>(1))
        {
            //m_data does not change

            //m_rows_idx: looking in m_cols_idx where there is an increase
            m_rows_idx.clear();
            m_rows_idx.reserve(m_nnz);
            for(std::size_t i = 1; i <= m_cols; ++i){ //looping from 1 since m_cols_idx has m_cols+1 elements, the first one being always 0
                if(m_cols_idx[i] > m_cols_idx[i-1]){
                    m_rows_idx.emplace_back(i-1);}}
            //m_cols_idx: only one column containing all the elements
            m_cols_idx.clear();
            m_cols_idx.reserve(2);
            m_cols_idx.emplace_back(static_cast<std::size_t>(0));
            m_cols_idx.emplace_back(m_nnz);
        }

        //col vector ==> row vector
        if(m_cols == static_cast<std::size_t>(1) && m_rows != static_cast<std::size_t>(1))
        {
            //m_data does not change

            //m_cols_idx: the old m_rows indeces indicates in which position there will be an increase by one in the new m_cols_idx
            m_cols_idx.clear();
            m_cols_idx.reserve(m_rows + 1);
            m_cols_idx.emplace_back(static_cast<std::size_t>(0));   //first element is always 0
            //loop solo sulle righe che ho, conto ogni quanto appare un elemento
            std::vector<std::size_t> rows_difference = m_rows_idx;
            rows_difference.push_back(m_rows);
            std::adjacent_difference(rows_difference.begin(),rows_difference.end(),rows_difference.begin());    //il primo elemento del risultato di std::adjacent_difference è sempre il primo elemento del range in input
            //the first element is always 0
            std::size_t el = m_cols_idx.back();
            for(auto it : rows_difference){
                //*it says how many elements have to be insert before increasing it
                for (std::size_t ii = 0; ii < it; ++ii){
                    m_cols_idx.emplace_back(el);}
                el += 1;}

            //m_rows_idx are all 0s, since all the elements will be in the first (and only) row
            m_rows_idx.clear();
            m_rows_idx.resize(m_nnz);
            std::fill(m_rows_idx.begin(),m_rows_idx.end(),static_cast<std::size_t>(0));
        }

        //general matrix ==> its transpost
        if(m_cols != static_cast<std::size_t>(1) && m_rows != static_cast<std::size_t>(1))
        {
            //new container for data
            std::vector< F_OBJ > temp_data;
            temp_data.reserve(this->size());
            //new container for row_idx
            std::vector<std::size_t> temp_row_idx;
            temp_row_idx.reserve(this->size());
            //new container for col_idx
            std::vector<std::size_t> temp_col_idx;
            temp_col_idx.reserve(this->rows() + 1);
            temp_col_idx.emplace_back(static_cast<std::size_t>(0));

            //loop su tutte le righe dell'ogetto originale, che diventeranno le colonne del trasposto
            for(std::size_t i = 0; i < this->rows(); ++i)
            {
                //conto quanti elementi ci sono nella vecchia riga i-th, quindi nuova colonna
                std::size_t counter_el_row_i = 0;
                //only if the row i-th is present
                if(this->check_row_presence(i))
                {
                    //trovo tutti gli elementi che sono nella riga i-th
                    for (std::size_t ii = 0; ii < m_nnz; ++ii)
                    {
                        //se l'elemento è nella riga i-th
                        if(m_rows_idx[ii] == i)
                        {   
                            //salvo(riga->col: sto salvando per colonne)
                            temp_data.emplace_back(m_data[ii]);
                            //un elemento in più in quella colonna
                            counter_el_row_i += static_cast<std::size_t>(1);
                            //come i nuovi indici di riga: la posizione ii mi indica dove salvo effettivamente l'elemento nel vettore
                            //dunque è l'elemento ii+1. Devo trovare il primo elemento negli indici di colonna che è >=(ii+1), e la sua posizione in m_cols_idx,
                            //tolto 1, mi dà la colonna in cui è salvato, dunque la riga nella trasposta
                            temp_row_idx.emplace_back(std::distance(m_cols_idx.cbegin(),std::lower_bound(m_cols_idx.cbegin(),m_cols_idx.cend(),ii+1) - 1));
                        }
                    }
                }
                //salvo quanti elementi in ogni colonna
                temp_col_idx.emplace_back(temp_col_idx.back() + counter_el_row_i);
            }
            //swap operations
            std::swap(m_data,temp_data);
            std::swap(temp_row_idx,m_rows_idx);
            std::swap(temp_col_idx,m_cols_idx);
        }
        //swap number of rows and cols
        std::swap(m_cols,m_rows);
    }

    /*!
    * @brief Tranpost of the functional sparse matrix 
    * @return a copy of the transpost 
    * @note does not transpose the original object
    */
    functional_matrix_sparse<INPUT,OUTPUT>
    transpose()
    const
    {
        functional_matrix_sparse<INPUT,OUTPUT> transpost_fm(*this);
        transpost_fm.transposing();

        return transpost_fm;
    }

    /*!
    * @brief Casting operator to a std::vector &, const version
    * @return a const reference to a std::vector of std::function, containing the function stored into the matrix
    * @code
    * functional_matrix_sparse<INPUT,OUTPUT>  fm;
    * std::vector< FUNC_OBJ<INPUT,OUTPUT> > & fm_v(fm); 
    * FUNC_OBJ<INPUT,OUTPUT> f = [](fm_utils::input_param_t<FUNC_OBJ<INPUT,OUTPUT>> x){return static_cast<OUTPUT>(10.0);};
    * fm_v.emplace_back(f);
    * @endcode
    */
    operator std::vector< F_OBJ > const &() const { return m_data; }

    /*!
    * @brief Casting operator to a std::vector &, non-const version
    * @return a reference to a std::vector of std::function, containing the function stored into the matrix
    */
    operator std::vector< F_OBJ > &() { return m_data; }
  
    /*!
    * @brief Casting to a std::vector &, const version
    * @return a const reference to a std::vector of std::function, containing the function stored into the matrix
    * @code
    * functional_matrix_sparse<INPUT,OUTPUT>  fm;
    * std::vector< FUNC_OBJ<INPUT,OUTPUT> > & fm_v{fm.sa_vector()}; 
    * FUNC_OBJ<INPUT,OUTPUT> f = [](fm_utils::input_param_t<FUNC_OBJ<INPUT,OUTPUT>> x){return static_cast<OUTPUT>(10.0);};
    * fm_v.emplace_back(f);
    * @endcode
    */
    std::vector< F_OBJ > const &
    as_vector() 
    const
    {
        return m_data;
    }

    /*!
    * @brief Casting to a std::vector &, non-const version
    * @return a const reference to a std::vector of std::function, containing the function stored into the matrix
    */
    std::vector< F_OBJ > &
    as_vector()
    {
        return m_data;
    }
};


/*!
* @brief Function to use range for loops over the stored functions, begin iterator
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @param fm a reference to a functional_matrix_sparse object
* @return an iterator to the begin of the container storing the std::function object
* @note use of declval in order to avoid to istantiate a vector to interrogate the type returned by begin
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
auto
begin(functional_matrix_sparse<INPUT,OUTPUT> &fm) 
    -> decltype(std::declval< std::vector<FUNC_OBJ<INPUT,OUTPUT>> >().begin())
{
  //exploiting the casting operator to std::vector<FUNC_OBJ<INPUT,OUTPUT>>&
  return static_cast<std::vector<FUNC_OBJ<INPUT,OUTPUT>> &>(fm).begin();
}

/*!
* @brief Function to use range for loops over the stored functions, end iterator
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @param fm a reference to a functional_matrix_sparse object
* @return an iterator to the end of the container storing the std::function object
* @note use of declval in order to avoid to istantiate a vector to interrogate the type returned by begin
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
auto
end(functional_matrix_sparse<INPUT,OUTPUT> &fm) 
    -> decltype(std::declval< std::vector<FUNC_OBJ<INPUT,OUTPUT>> >().end())
{
  //exploiting the casting operator to std::vector<FUNC_OBJ<INPUT,OUTPUT>>&
  return static_cast<std::vector<FUNC_OBJ<INPUT,OUTPUT>> &>(fm).end();
}

/*!
* @brief Function to use range for loops over the stored functions, const begin iterator
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @param fm a const reference to a functional_matrix_sparse object
* @return a constant iterator to the begin of the container storing the std::function object
* @note use of declval in order to avoid to istantiate a vector to interrogate the type returned by begin
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
auto
cbegin(functional_matrix_sparse<INPUT,OUTPUT> const &fm)
  -> decltype(std::declval< std::vector<FUNC_OBJ<INPUT,OUTPUT>> >().cbegin())
{
  //exploiting the casting operator to std::vector<FUNC_OBJ<INPUT,OUTPUT>>& const
  return static_cast<std::vector<FUNC_OBJ<INPUT,OUTPUT>> const &>(fm).cbegin();
}

/*!
* @brief Function to use range for loops over the stored functions, const end iterator
* @tparam INPUT type of abscissa
* @tparam OUTPUT type of image
* @param fm a const reference to a functional_matrix_sparse object
* @return a constant iterator to the end of the container storing the std::function object
* @note use of declval in order to avoid to istantiate a vector to interrogate the type returned by begin
*/
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
auto
cend(functional_matrix_sparse<INPUT,OUTPUT> const &fm) 
    -> decltype(std::declval< std::vector<FUNC_OBJ<INPUT,OUTPUT>> >().cend())
{
  //exploiting the casting operator to std::vector<FUNC_OBJ<INPUT,OUTPUT>>& const
  return static_cast<std::vector<FUNC_OBJ<INPUT,OUTPUT>> const &>(fm).cend();
}

#endif  /*FUNCTIONAL_MATRIX_SPARSE_HPP*/