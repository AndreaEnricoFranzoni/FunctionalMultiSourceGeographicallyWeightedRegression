// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __FDAPDE_PARSING_H__
#define __FDAPDE_PARSING_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

// A collection of convinient parsing utils

template <typename CharBuff>
concept is_char_buff =
  requires(CharBuff c, std::size_t i) {
      { c.operator[](i) } -> std::convertible_to<char>;
  } || (std::is_pointer_v<CharBuff> && std::is_same_v<std::remove_cvref_t<CharBuff>, std::add_const_t<char>*>) ||
  (std::is_array_v<std::remove_cvref_t<CharBuff>> &&
   std::is_same_v <std::remove_cvref_t<decltype(std::declval<CharBuff>()[0])>, char>);

// returns the position of the next occurence of c in buff, buff_sz if no occurence is found
template <typename CharBuff>
    requires(is_char_buff<CharBuff>)
size_t next_char_(const CharBuff& buff, std::size_t begin, std::size_t end, char c) {
    std::size_t i = begin;
    while (i < end && buff[i] != c) { i++; }
    return i - begin;
}
// returns the position of the next occurence of c or '\n' in buff, end if no occurence is found
template <typename CharBuff>
    requires(is_char_buff<CharBuff>)
size_t next_char_or_newline_(const CharBuff& buff, std::size_t begin, std::size_t end, char c) {
    std::size_t i = begin;
    while (i < end && buff[i] != c && (buff[i] != EOF && buff[i] != '\n' && buff[i] != '\r')) { i++; }
    return i - begin;
}

// transforms a char buffer into a stream of tokens separated by sep
template <typename CharT> struct token_stream {
    using buff_t = std::add_pointer_t<CharT>;
    using size_t = std::size_t;

    token_stream() = default;
    token_stream(CharT* buff, size_t buff_sz, char sep) :
        buff_(buff), sep_(sep), head_(0), tail_(0), buff_sz_(buff_sz) { }
    token_stream(const std::string& buff, char sep) :
        buff_(buff.c_str()), sep_(sep), head_(0), tail_(0), buff_sz_(buff.size()) { }

    // a line is a contiguous portion of buffer encolsed between newline chars '\n'
    struct line_iterator {
        using value_t = std::string_view;
        using reference = std::add_lvalue_reference_t<value_t>;
        using pointer = std::add_pointer_t<value_t>;
      
        line_iterator() noexcept = default;
        line_iterator(buff_t buff, size_t buff_sz, size_t begin, size_t end, char sep) :
            buff_(buff), sep_(sep), token_sz_(0), buff_sz_(buff_sz), begin_(begin), end_(end), pos_(0) {
            fetch_token_();
        }
        bool has_token() const { return has_token_; }
        operator bool() { return has_token(); }
        line_iterator& operator++() {
            fetch_token_();
            return *this;
        }
        reference get_token() { return token_; }
        bool eol() const { return pos_ >= (end_ - begin_); }     // true if token_ is the last of this line
        bool eof() const { return begin_ + pos_ >= buff_sz_; }   // true if token_ is the last of the stream
        size_t n_tokens() {
            size_t n = 0;
	    pos_ = 0;
            while (has_token()) {
                n++;
                fetch_token_();
            }
            // reset status
            token_sz_ = 0;
            pos_ = 0;
	    token_ = std::string_view{};
            return n;
        }
       private:
        void fetch_token_() {
            has_token_ = pos_ < (end_ - begin_);
            if (has_token_) {
                token_sz_ = next_char_or_newline_(buff_, begin_ + pos_, end_, sep_);
                token_ = value_t(buff_ + (begin_ + pos_), token_sz_);
                pos_ += token_sz_ + 1;
            }
        }
        value_t token_;
        const buff_t buff_;
        char sep_;
        size_t token_sz_, buff_sz_;
        size_t begin_, end_, pos_;
        mutable bool has_token_ = true;
    };
    // observers
    line_iterator get_line() {
        head_ = tail_;
        tail_ = tail_ + next_char_(buff_, head_, buff_sz_, '\n');
        return line_iterator {buff_, buff_sz_, head_, tail_++, sep_};
    }
    bool has_line() const { return head_ < buff_sz_; }
    operator bool() const { return head_ < buff_sz_; }
   private:
    const buff_t buff_;
    char sep_;
    size_t head_, tail_;
    size_t buff_sz_;
};

// double parsing function handling only decimal point ([0-9]^+.[0-9]^+) and exponential ([0-9]^+.[0-9]^+e[+/-][0-9]^+)
// floating point formats (faster than std::stod())
template <typename CharBuff>
    requires(is_char_buff<CharBuff>)
double stod(CharBuff&& str) {
    auto is_na = [](CharBuff& chr, int& i) -> bool {
        if (chr[i] == 'N') {
            i++;
	    char c = chr[i];
            if (c == 'A') {
                i += 1;
                return true;
            }
            if (c == 'a' && chr[i + 1] == 'N') {
                i += 2;
                return true;
            }
	    i--;
            return false;
        }
        if (chr[i] == 'n' && chr[i + 1] == 'a' && chr[i + 2] == 'n') {
            i += 3;
            return true;
        }
        return false;
    };

    double val = 0;
    int i = 0;
    while (str[i] == ' ') { i++; }   // skip leading whitespaces
    if (is_na(str, i)) { return std::numeric_limits<double>::quiet_NaN(); }
    if (!(str[i] == '-' || (str[i] >= '0' && str[i] <= '9'))) { throw std::invalid_argument("stod parsing error."); }
    int sign = 1;
    if (str[i] == '-') {
        sign = -1;
        i++;
    }
    // parse integer part
    while (str[i] >= '0' && str[i] <= '9') {
        val = val * 10 + (str[i] - '0');
        i++;;
    }
    bool maybe_scientific = val < 10;
    if (str[i] == '.') {   // expect the decimal point
        i++;;
        double dec = 0.1;
        while (str[i] >= '0' && str[i] <= '9') {
            val = val + (str[i] - '0') * dec;
            dec *= 0.1;
	    i++;;
        }
        if (maybe_scientific && (str[i] == 'e' || str[i] == 'E')) {   // scientific notation parsing
            i++;;
            int exp_sign = (str[i] == '-') ? -1 : +1;
            i++;;
            int exp = 0;
            while (str[i] >= '0' && str[i] <= '9') {
                exp = exp * 10 + (str[i] - '0');
                i++;;
            }
            val *= (exp_sign > 0) ? std::pow(10, exp) : std::pow(0.1, exp);
        }
    }
    return sign * val;
}

// integer parsing function
template <typename CharBuff>
    requires(is_char_buff<CharBuff>)
int stoi(CharBuff&& str) {
    int val = 0;
    int i = 0;
    while (str[i] == ' ') { i++; }   // skip leading whitespaces
    if (!(str[i] == '-' || (str[i] >= '0' && str[i] <= '9'))) { throw std::invalid_argument("stoi parsing error."); }
    int sign = 1;
    if (str[i] == '-') {
        sign = -1;
        i++;
    }
    while (str[i] >= '0' && str[i] <= '9') {
        val = val * 10 + (str[i] - '0');
        i++;;
    }
    return sign * val;
}

// reader for table of values of type T
template <typename T> class table_reader {
   private:
    template <typename CharBuff>
        requires(internals::is_char_buff<CharBuff>)
    T parse_value_(const CharBuff& token) const {
        // check if token is recognized as na
        if (std::find(na_values_.begin(), na_values_.end(), token) != na_values_.end()) {
            return std::numeric_limits<T>::quiet_NaN();
        }
	// parse token as numeric
        if constexpr (std::is_same_v<T, double>) { return internals::stod(token); }
        if constexpr (std::is_same_v<T, int   >) { return internals::stoi(token); }
        return T {};
    }

    std::string_view& skipquote_(bool skip_quote, std::string_view& token) const {
        if (skip_quote) { [[likely]]
            if (!token.empty() && token.front() == '"') token.remove_prefix(1);
            if (!token.empty() && token.back()  == '"') token.remove_suffix(1);
        }
        return token;
    }
    // parsed data
    std::vector<T> data_ {};
    std::size_t n_cols_ = 0, n_rows_ = 0;
    std::vector<std::string> colnames_ {};

    std::vector<std::string> na_values_ = {"NA", "NaN", "nan"};
   public:
    table_reader() = default;
    table_reader(
      const char* filename, bool header, char sep, bool index_col, bool skip_quote = true, std::size_t chunksize = 4) :
        n_cols_(0), n_rows_(0), colnames_() {
        parse(filename, header, sep, index_col, skip_quote, chunksize);
    }
    table_reader(const char* filename, bool index_col, bool skip_quote = true, std::size_t chunksize = 4) :
        table_reader(filename, true, ',', index_col, skip_quote, chunksize) { }
    table_reader(const std::string& filename, bool index_col, bool skip_quote = true, std::size_t chunksize = 4) :
        table_reader(filename.c_str(), index_col, skip_quote, chunksize) { }

    // observers
#ifdef __FDAPDE_HAS_EIGEN__
    Eigen::Map<const Eigen::Matrix<T, Dynamic, Dynamic, Eigen::RowMajor>> as_matrix() const {
        return Eigen::Map<const Eigen::Matrix<T, Dynamic, Dynamic, Eigen::RowMajor>>(data_.data(), n_rows_, n_cols_);
    }
#endif

    // extract data by column name
    std::vector<T> col(const std::string& colname) {
        std::vector<T> col_(n_rows_);
        int i = 0;
        {
            std::string cmp = "";
            for (; i < n_cols_ && cmp != colname; ++i) { cmp = colnames_[i]; }
	    fdapde_assert(i < n_cols_ && cmp == colname);
        }
        for (int j = 0; j < n_rows_; ++j) { col_[i] = data_[i + j * n_cols_]; }
	return col_;
    }
    // modifiers
    void set_na_values(const std::vector<std::string>& na_values) { na_values_ = na_values; }
    std::size_t cols() const { return n_cols_; }
    std::size_t rows() const { return n_rows_; }
    const std::vector<T>& data() const { return data_; }
    const std::vector<std::string>& colnames() const { return colnames_; }
    // parsing function
    void parse(
      const char* filename, bool header = true, char sep = ',', bool index_col = true, bool skip_quote = true,
      std::size_t chunksize = 4) {
        std::string filename_ = std::filesystem::current_path().string() + "/" + filename;
        if (!std::filesystem::exists(filename_))
            throw std::runtime_error("file " + std::string(filename_) + " not found.");
        auto stream = batched_istream(filename_, chunksize); 
        bool header_ = header;
	std::size_t col_id = 0;
        std::string last_token;
	std::size_t n_file_cols = 0;

        while (stream) {
            stream.read();
            const char* buff = stream.data();
            // tokenize input stream
            internals::token_stream token_stream_(buff, stream.size(), sep);

	    // TODO: bug when file doesn't fit in chunksize
	    
            while (token_stream_) {
                auto line = token_stream_.get_line();
                if (header_) { [[unlikely]]   // header parsing logic
                    header_ = false;
                    while (line.has_token()) {
                        std::string_view& token = skipquote_(skip_quote, line.get_token());
                        if (index_col == true) {
                            if (n_file_cols != 0) colnames_.push_back(std::string(token));
                        } else {
                            colnames_.push_back(std::string(token));
                        }
                        n_file_cols++;
                        ++line;
                    }
		    n_cols_ = n_file_cols - (index_col == true ? 1 : 0);
                } else {   // data parsing logic
                    while (line.has_token()) {
                        if (index_col == true && col_id == 0) {   // skip first column
                        } else {
                            std::string_view& token = skipquote_(skip_quote, line.get_token());
                            if (line.eof()) {   // skip parsing and wait for next block
                                last_token = token;
                            } else {
                                if (!last_token.empty()) {
                                    last_token = last_token + std::string(token);   // merge tokens
                                    data_.push_back(parse_value_(last_token));
                                    last_token.clear();
                                } else if (!token.empty()) {
                                    data_.push_back(parse_value_(token));
                                }
                            }
                        }
                        if (!line.eof() ) { col_id = (col_id + 1) % n_file_cols; }
                        ++line;
                    }
                }
            }
        }
        // process evantual last token of the last block of the stream
        if (!last_token.empty()) { data_.push_back(parse_value_(last_token)); }
        if (data_.size() % n_cols_ != 0) throw std::invalid_argument("parsing error.");
        n_rows_ = data_.size() / n_cols_;
        return;
    }
};

}   // namespace internals  
}   // namespace fdapde

#endif   // __FDAPDE_PARSING_H__
