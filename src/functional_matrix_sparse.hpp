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


#ifndef FUNCTIONAL_MATRIX_SPARSE_HPP
#define FUNCTIONAL_MATRIX_SPARSE_HPP


#include "functional_matrix_expression_wrapper.hpp"
#include "functional_matrix_utils.hpp"
#include <utility>
#include <vector>
#include <cassert>

#include <iostream>


//! A class for sparse matrices of functions, in compress format: STORING COLUMN WISE (CSC)
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
class functional_matrix_sparse : public Expr< functional_matrix_sparse<INPUT,OUTPUT>, INPUT, OUTPUT >
{
    //type of the function stored
    using F_OBJ = FUNC_OBJ<INPUT,OUTPUT>;
    using F_OBJ_INPUT = fm_utils::input_param_t<F_OBJ>;

    //null function (const version)
    inline static const F_OBJ m_null_function = [](F_OBJ_INPUT x){ return static_cast<OUTPUT>(0);};
    //null function (non-const verion)
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
    /*!Vector that contains the non-null functions. Size is nnz.*/
    std::vector< F_OBJ > m_data;    

public:
    //! default constructor
    functional_matrix_sparse() = default;
    //! A vector may be converted to a Vector
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
    //! Move constructor
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
    //! Copy constructor
    functional_matrix_sparse(functional_matrix_sparse const &) = default;
    //! Move constructor
    functional_matrix_sparse(functional_matrix_sparse &&) = default;
    //! Copy assign
    functional_matrix_sparse &operator=(functional_matrix_sparse const &) = default;
    //! Move assign
    functional_matrix_sparse &operator=(functional_matrix_sparse &&) = default;


    //! I may build a Vector from an expression!    TODO
    template <class T> 
    functional_matrix_sparse(const Expr<T,INPUT,OUTPUT> &e)
        :   m_data()
    {
        const T &et(e); // casting!
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
                //il confronto con funzioni è un casino. Siccome non posso confrontare l'indirizzo direttamente (&et(i,j))
                //(non prende una non-const ref da un temporaneo o da un const)
                //inserisco, confronto, e poi tolgo: NON FUNZIONA
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

    //! Assigning an expression                     TODO
    /*!
        This method is fundamental for expression template technique.
    */
    template <class T>
    functional_matrix_sparse &
    operator=(const Expr<T,INPUT,OUTPUT> &e)
    {
        const T &et(e); // casting!
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
                if(&m_data.back() == &functional_matrix_sparse<INPUT,OUTPUT>::m_null_function_non_const)    // NON FUNZIONA
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



    //AUXILIARY FUNCTIONS FOR CHECKING THE PRESENCES
    /*!
    * @brief Checking presence of row idx-th: at least a nnz element in that row.
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
    * @brief Checking presence of col idx-th: at least a nnz element in that col.
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
    * @brief Checking for the presence of the element in position (i,j).
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
    * @brief Returns element (i,j) (being the sparse matrix compressed, it is possible to modify only element already there)
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
    * @brief Returns element (i,j) (const version) 
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
    */
    std::size_t
    rows() 
    const
    {
        return m_rows;
    }

    /*!
    * @brief Cols size
    */
    std::size_t
    cols() 
    const
    {
        return m_cols;
    }

    /*!
    * @brief Number of elements
    */
    std::size_t
    size() 
    const
    {
        return m_nnz;
    }

    /*!
    * @brief Row indices of the non-zero elements
    */
    std::vector<std::size_t>
    rows_idx()
    const
    {
        return m_rows_idx;
    }

    /*!
    * @brief Cumulative number of elements up to col i-th
    */
    std::vector<std::size_t>
    cols_idx()
    const
    {
        return m_cols_idx;
    }

    /*!
    * @brief Transposing the functional matrix
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
            std::vector<std::size_t> new_cols_idx{0,m_nnz};
            m_cols_idx = new_cols_idx;

                    std::cout << "Dentro il branch row==>col" << std::endl;
        std::cout << "Indici di riga" << std::endl;
        for(std::size_t i=0; i < m_rows_idx.size(); ++i){std::cout<<m_rows_idx[i]<<std::endl;}
            std::cout << "Indici di colonna" << std::endl;
        for(std::size_t i=0; i < m_cols_idx.size(); ++i){std::cout<<m_cols_idx[i]<<std::endl;}
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

        std::cout << "Fine trasposto: " << m_rows << " righe e " << m_cols << " colonne" << std::endl;
        std::cout << "Indici di riga" << std::endl;
        for(std::size_t i=0; i < m_rows_idx.size(); ++i){std::cout<<m_rows_idx[i]<<std::endl;}
            std::cout << "Indici di colonna" << std::endl;
        for(std::size_t i=0; i < m_cols_idx.size(); ++i){std::cout<<m_cols_idx[i]<<std::endl;}
    }

    /*!
    * @brief Tranpost of the functional matrix (copy of it)
    */
    functional_matrix_sparse<INPUT,OUTPUT>
    transpose()
    const
    {
        functional_matrix_sparse<INPUT,OUTPUT> transpost_fm(*this);
        transpost_fm.transposing();

        return transpost_fm;
    }

    //! May be cast to a std::vector &
    /*!
    This way I can use all the methods of a std::vector!
    @code
    Vector a;
    std::vector<double> & av(a); // you cannot use {a} here!
    av.emplace_back(10.0);
    @endcode
    */
    operator std::vector< F_OBJ > const &() const { return m_data; }
    //! Non const version of casting operator
    operator std::vector< F_OBJ > &() { return m_data; }
  
    //! This does the same as casting but with a method
    /*!
    Only to show that you do not need casting operators, if you find them
    confusing (or if the compiler gets confused!)
    @code
    Vector a;
    std::vector<double> & av{a.as_vector()}; // Here you may use {}! (but ()
    works as well!) av.emplace_back(10.0);
    @endcode
    */
    std::vector< F_OBJ > const &
    as_vector() 
    const
    {
        return m_data;
    }
    //! Non const version of casting operator
    std::vector< F_OBJ > &
    as_vector()
    {
        return m_data;
    }
};


//! I want to use range for loops with Vector objects.
/*!
  Note the use of declval. I do not need to istantiate a vector to interrogate
  the type returned by begin!
 */
template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
auto
begin(functional_matrix_sparse<INPUT,OUTPUT> &fm) 
    -> decltype(std::declval< std::vector<FUNC_OBJ<INPUT,OUTPUT>> >().begin())
{
  // I exploit the fact tha I have a casting operator to std::vector<double>&
  return static_cast<std::vector<FUNC_OBJ<INPUT,OUTPUT>> &>(fm).begin();
  // If you prefer
  // return fm.as_vector().begin();
}

template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
auto
end(functional_matrix_sparse<INPUT,OUTPUT> &fm) 
    -> decltype(std::declval< std::vector<FUNC_OBJ<INPUT,OUTPUT>> >().end())
{
  // I exploit the fact tha I have a casting operator to std::vector<double>&
  return static_cast<std::vector<FUNC_OBJ<INPUT,OUTPUT>> &>(fm).end();
}

template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
auto
cbegin(functional_matrix_sparse<INPUT,OUTPUT> const &fm)
  -> decltype(std::declval< std::vector<FUNC_OBJ<INPUT,OUTPUT>> >().cbegin())
{
  // I exploit the fact tha I have a casting operator to std::vector<double>
  // const &
  return static_cast<std::vector<FUNC_OBJ<INPUT,OUTPUT>> const &>(fm).cbegin();
}

template< typename INPUT = double, typename OUTPUT = double >
    requires (std::integral<INPUT> || std::floating_point<INPUT>)  &&  (std::integral<OUTPUT> || std::floating_point<OUTPUT>)
inline 
auto
cend(functional_matrix_sparse<INPUT,OUTPUT> const &fm) 
    -> decltype(std::declval< std::vector<FUNC_OBJ<INPUT,OUTPUT>> >().cend())
{
  // I exploit the fact tha I have a casting operator to std::vector<double>
  // const &
  return static_cast<std::vector<FUNC_OBJ<INPUT,OUTPUT>> const &>(fm).cend();
}

#endif  /*FUNCTIONAL_MATRIX_SPARSE_HPP*/