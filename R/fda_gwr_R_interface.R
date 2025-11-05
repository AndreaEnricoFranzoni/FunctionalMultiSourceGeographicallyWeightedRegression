#' @title installation_fdagwr
#' @name installation_fdagwr
#' @description
#' Check **`fdagwr`** package installation
#' @export
#' @author Andrea Enrico Franzoni
NULL



#' @title FMSGWR_ESC
#' @name FMSGWR_ESC
#' @description
#' Fitting a Functional Multi-Source Geographically Weighted Regression ESC model. The covariates are functional objects, divided into
#' three categories: stationary covariates (C), constant over geographical space, event-dependent covariates (E), that vary depending on the spatial coordinates of the event, 
#' station-dependent covariates (S), that vary depending on the spatial coordinates of the stations that measure the event. Regression coefficients are estimated 
#' in the following order: C, S, E. The functional response is already reconstructed according to the method proposed by Bortolotti et Al. (2024)
#' @param y_points **`numeric matrix`** of double containing the raw response: each row represents a specific abscissa for which the response evaluation is available, each column a statistical unit. Response is a already reconstructed.
#' @param t_points **`numeric vector`** of double with the abscissa points with respect of the raw evaluations of \p y_points are available (length of \p t_points is equal to the number of rows of \p y_points).
#' @param left_extreme_domain **`double`** indicating the left extreme of the functional data domain (not necessarily the smaller element in \p t_points).
#' @param right_extreme_domain double indicating the right extreme of the functional data domain (not necessarily the biggest element in \p t_points).
#' @param coeff_y_points **`numeric matrix`** of double containing the coefficient of response's basis expansion: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
#' @param knots_y_points **`numeric vector`** of double with the abscissa points with respect which the basis expansions of the response and response reconstruction weights are performed (all elements contained in [a,b]). 
#' @param degree_basis_y_points **`non-negative integer`**: the degree of the basis used for the basis expansion of the (functional) response. Default explained below (can be **`NULL`**).
#' @param degree_basis_y_points **`non-negative integer`**: the degree of the basis used for the basis expansion of the (functional) response. Default explained below (can be **`NULL`**).
#' @param n_basis_y_points **`positive integer`**: number of basis for the basis expansion of the (functional) response. It must match number of rows of coeff_y_points. Default explained below (can be **`NULL`**).
#' @param n_basis_y_points **`positive integer`**: number of basis for the basis expansion of the (functional) response. It must match number of rows of coeff_y_points. Default explained below (can be **`NULL`**).
#' @param coeff_rec_weights_y_points **`numeric matrix`** of double containing the coefficients of the basis expansion of the weights to reconstruct the (functional) response: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
#' @param degree_basis_rec_weights_y_points **`non-negative integer`**: the degree of the basis used for response reconstruction weights. Default explained below (can be **`NULL`**).
#' @param n_basis_rec_weights_y_points **`positive integer`**: number of basis for the basis expansion of response reconstruction weights. It must match number of rows of \p coeff_rec_weights_y_points. Default explained below (can be **`NULL`**).
#' @param coeff_stationary_cov **`list of numeric matrices`** of double: element i-th containing the coefficients for the basis expansion of the i-th stationary covariate: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
#'                             The name of the i-th element is the name of the i-th stationary covariate (default: **`"reg.Ci"`** if no name present).
#' @param basis_types_stationary_cov **`vector of strings`**, element i-th containing the type of basis used for the i-th stationary covariate basis expansion. Possible values: **`"bsplines"`**, **`"constant"`**. Defalut: **`"bsplines"`**.
#' @param knots_stationary_cov **`numeric vector`** of double with the abscissa points with respect which the basis expansions of the stationary covariates are performed (all elements contained in [a,b]). 
#' @param degrees_basis_stationary_cov **`integer vector`** of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th stationary covariate. Default explained below (can be **`NULL`**)).
#' @param n_basis_stationary_cov **`integer vector`** of positive integers: element i-th is the number of basis for the basis expansion of the i-th stationary covariate. It must match number of rows of the i-th element of \p coeff_stationary_cov. Default explained below (can be **`NULL`**).
#' @param penalization_stationary_cov **`numeric vector`** of non-negative double: element i-th is the penalization used for the i-th stationary covariate.
#' @param knots_beta_stationary_cov **`numeric vector`** of double with the abscissa points with respect which the basis expansions of the stationary covariates functional regression coefficients are performed (all elements contained in [a,b]).
#' @param degrees_basis_beta_stationary_cov **`integer vector`**  of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th stationary covariate functional regression coefficients. Default explained below (can be **`NULL`**).
#' @param n_basis_beta_stationary_cov **`integer vector`** of positive integers: element i-th is the number of basis for the basis expansion of the i-th stationary covariate functional regression coefficients. Default explained below (can be **`NULL`**).
#' @param coeff_events_cov **`list of numeric matrices`** of double: element i-th containing the coefficients for the basis expansion of the i-th events-dependent covariate: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
#'                         The name of the i-th element is the name of the i-th events-dependent covariate (default: **`"reg.Ei"`** if no name present).
#' @param basis_types_events_cov **`vector of strings`**, element i-th containing the type of basis used for the i-th events-dependent covariate basis expansion. Possible values: **`"bsplines"`**, **`"constant"`**. Defalut: **`"bsplines"`**.
#' @param knots_events_cov **`numeric vector`** of double with the abscissa points with respect which the basis expansions of the events-dependent covariates are performed (all elements contained in [a,b]).                        
#' @param degrees_basis_events_cov **`integer vector`** of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th events-dependent covariate. Default explained below (can be **`NULL`**).
#' @param n_basis_events_cov **`integer vector`** of positive integers: element i-th is the number of basis for the basis expansion of the i-th events-dependent covariate. It must match number of rows of the i-th element of \p coeff_events_cov. Default explained below (can be **`NULL`**).
#' @param penalization_events_cov **`numeric vector`** of non-negative double: element i-th is the penalization used for the i-th events-dependent covariate.
#' @param coordinates_events **`numeric matrix`** of double containing the UTM coordinates of the event of each statistical unit: each row represents a statistical unit, each column a coordinate (2 columns).
#' @param kernel_bandwith_events **`positive double`** indicating the bandwith of the gaussian kernel used to smooth the distances within events.
#' @param knots_beta_events_cov **`numeric vector`** of double with the abscissa points with respect which the basis expansions of the events-dependent covariates functional regression coefficients are performed (all elements contained in [a,b]). 
#' @param degrees_basis_beta_events_cov **`integer vector`** of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th events-dependent covariate functional regression coefficient. Default explained below (can be **`NULL`**). 
#' @param n_basis_beta_events_cov **`integer vector`** of positive integers: element i-th is the number of basis for the basis expansion of the i-th events-dependent covariate functional regression coefficient. Default explained below (can be **`NULL`**).
#' @param coeff_stations_cov **`list of numeric matrices`** of double: element i-th containing the coefficients for the basis expansion of the i-th stations-dependent covariate: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit.
#'                           The name of the i-th element is the name of the i-th stations-dependent covariate (default: **`"reg.Si"`** ).
#' @param basis_types_stations_cov **`vector of strings`**, element i-th containing the type of basis used for the i-th stations-dependent covariates basis expansion. Possible values: **`"bsplines"`**, **`"constant"`**. Defalut: **`"bsplines"`**.
#' @param knots_stations_cov **`numeric vector`** of double with the abscissa points with respect which the basis expansions of the stations-dependent covariates are performed (all elements contained in [a,b]). 
#' @param degrees_basis_stations_cov **`integer vector`** of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th stations-dependent covariate. Default explained below (can be **`NULL`**).
#' @param n_basis_stations_cov **`integer vector`** of positive integers: element i-th is the number of basis for the basis expansion of the i-th stations-dependent covariate. It must match number of rows of the i-th element of \p coeff_stations_cov. Default explained below (can be **`NULL`**).
#' @param penalization_stations_cov **`numeric vector`** of non-negative double: element i-th is the penalization used for the i-th stations-dependent covariate.
#' @param coordinates_stations **`numeric matrix`** of double containing the UTM coordinates of the station of each statistical unit: each row represents a statistical unit, each column a coordinate (2 columns).
#' @param kernel_bandwith_stations **`positive double`** indicating the bandwith of the gaussian kernel used to smooth the distances within stations.
#' @param knots_beta_stations_cov **`numeric vector`** of double with the abscissa points with respect which the basis expansions of the stations-dependent covariates functional regression coefficients are performed (all elements contained in [a,b]). 
#' @param degrees_basis_beta_stations_cov **`integer vector`** of non-negative integers: element i-th is the degree of the basis used for the basis expansion of the i-th stations-dependent covariate functional regression coefficient. Default explained below (can be **`NULL`**).
#' @param n_basis_beta_stations_cov **`integer vector`** of positive integers: element i-th is the number of basis for the basis expansion of the i-th stations-dependent covariate functional regression coefficient. Default explained below (can be **`NULL`**).
#' @param in_cascade_estimation **`bool`**: if false, an exact algorithm taking account for the interaction within non-stationary covariates is used to fit the model. Otherwise, the model is fitted in cascade. The first option is more precise, but way more computationally intensive.
#' @param n_knots_smoothing **`positive integer`**: number of knots used to perform the smoothing on the response obtained leaving out all the non-stationary components (default: **`100`**).
#' @param n_intervals_quadrature **`positive integer`**: number of intervals used while performing integration via midpoint (rectangles) quadrature rule (default: **`100`**).
#' @param num_threads **`positive integer`**: number of threads to be used in OMP parallel directives. Default: **`NULL`**, that is equivalent to the maximum number of cores available in the machine.
#' @param basis_type_y_points **`string`** containing the type of basis used for the functional response basis expansion. Possible values: **`"bsplines"`**, **`"constant"`**. Defalut: **`"bsplines"`**.
#' @param basis_type_rec_weights_y_points **`string`** containing the type of basis used for the weights to reconstruct the functional response basis expansion. Possible values: **`"bsplines"`**, **`"constant"`**. Defalut: **`"bsplines"`**.
#' @param basis_types_beta_stationary_cov **`vector of strings`**, element i-th containing the type of basis used for the i-th stationary covariate functional regression coefficients basis expansion. Possible values: **`"bsplines"`**, **`"constant"`**. Defalut: **`"bsplines"`**.
#' @param basis_types_beta_events_cov **`vector of strings`**, element i-th containing the type of basis used for the i-th events-dependent covariate functional regression coefficients basis expansion. Possible values: **`"bsplines"`**, **`"constant"`**. Defalut: **`"bsplines"`**.
#' @param basis_types_beta_stations_cov **`vector of strings`**, element i-th containing the type of basis used for the i-th stations-dependent covariate functional regression coefficients basis expansion. Possible values: **`"bsplines"`**, **`"constant"`**. Defalut: **`"bsplines"`**.
#' @return **`list`** containing:
#'                   \itemize{
#'                   \item 'FGWR': **`string`**: the type of fgwr used (**`"FMSGWR_ESC"`**);
#'                   \item 'EstimationTechnique': **`string`**: **`"Exact"`** if \p in_cascade_estimation false, **`"Cascade"`** if \p in_cascade_estimation true 
#'                   \item 'Bc': **`list`** containing, for each stationary covariate regression coefficient (each element is named with the element names in the list \p coeff_stationary_cov (default, if not given: **`"CovC*"`**)):
#'                         \itemize{
#'                         \item 'basis_coeff': **`numeric vector`** of double, dimension Lc_jx1,containing the coefficients of the basis expansion of the beta.
#'                         \item 'basis_type': **`string`** containing the basis type over which the beta basis expansion is performed. Possible values: **`"bsplines"`**, **`"constant"`**. (Respective element of \p basis_types_beta_stationary_cov).
#'                         \item 'basis_num': **`positive integer`**: the number of basis used for performing the beta basis expansion (respective elements of \p n_basis_beta_stationary_cov).
#'                         \item 'knots': **`numeric vector`** of double: knots used to create the basis system for the beta (it is the input \p knots_beta_stationary_cov).
#'                         }
#'                   \item 'Beta_c': **`list`** containing, for each stationary covariate regression coefficient (each element is named with the element names in the list coeff_stationary_cov (default, if not given: **`"CovC*"`**)) a list with:
#'                         \itemize{
#'                         \item 'Beta_eval': **`numeric vector`** of double containing the discrete evaluations of the stationary beta.
#'                         \item 'Abscissa': **`numeric vector`** of double containing the domain points for which the evaluation of the beta is available (it is the input \p t_points).
#'                         }
#'                   \item 'Be': **`list`** containing, for each event-dependent covariate regression coefficient (each element is named with the element names in the list \p coeff_events_cov (default, if not given: **`"CovE*"`**)):
#'                         \itemize{
#'                         \item 'basis_coeff': **`list`**, containing, for each statistical unit, a **`numeric vector`** of double, dimension Le_jx1,containing the coefficients of the basis expansion of the beta for that unit.
#'                         \item 'basis_type': **`string`** containing the basis type over which the beta basis expansion is performed. Possible values: **`"bsplines"`**, **`"constant"`**. (Respective element of \p basis_types_beta_events_cov).
#'                         \item 'basis_num': **`positive integer`**: the number of basis used for performing the beta basis expansion (respective elements of \p n_basis_beta_events_cov).
#'                         \item 'knots': **`numeric vector`** of double: knots used to create the basis system for the beta (it is the input \p knots_beta_events_cov).
#'                         }
#'                   \item 'Beta_e': **`list`** containing, for each event-dependent covariate regression coefficient (each element is named with the element names in the list \p coeff_events_cov (default, if not given: **`"CovE*"`**)) a list with:
#'                         \itemize{
#'                         \item 'Beta_eval': **`list`**, containing, for each statistical unit, a **`numeric vector`** of double containing the discrete evaluations of the non-stationary beta, one for each unit.
#'                         \item 'Abscissa': **`numeric vector`** of double containing the domain points for which the evaluation of the beta is available (it is the input \p t_points).
#'                         }
#'                   \item 'Bs': **`list`** containing, for each station-dependent covariate regression coefficient (each element is named with the element names in the list \p coeff_stations_cov (default, if not given: **`"CovS*"`**)):
#'                         \itemize{
#'                         \item 'basis_coeff': **`list`**, containing, for each statistical unit, a **`numeric vector`** of double, dimension Ls_jx1,containing the coefficients of the basis expansion of the beta for that unit.
#'                         \item 'basis_type': **`string`** containing the basis type over which the beta basis expansion is performed. Possible values: **`"bsplines"`**, **`"constant"`**. (Respective element of \p basis_types_beta_stations_cov).
#'                         \item 'basis_num': **`positive integer`**: the number of basis used for performing the beta basis expansion (respective elements of \p n_basis_beta_stations_cov).
#'                         \item 'knots': **`numeric vector`** of double: knots used to create the basis system for the beta (it is the input \p knots_beta_stations_cov).
#'                         }
#'                   \item 'Beta_e': **`list`** containing, for each station-dependent covariate regression coefficient (each element is named with the element names in the list \p coeff_stations_cov (default, if not given: **`"CovS*"`**)) a list with:
#'                         \itemize{
#'                         \item 'Beta_eval': **`list`**, containing, for each statistical unit, a **`numeric vector`** of double containing the discrete evaluations of the non-stationary beta, one for each unit.
#'                         \item 'Abscissa': **`numeric vector`** of double containing the domain points for which the evaluation of the beta is available (it is the input \p t_points).
#'                         }
#'                   \item 'predictor_info': **`list`** containing partial residuals and information of the fitted model to perform predictions for new statistical units:
#'                         \itemize{
#'                         \item 'partial_res': **`list`** containing information to compute the partial residuals:
#'                               \itemize{
#'                               \item 'c_tilde_hat': **`numeric vector`** of double with the basis expansion coefficients of the response minus the stationary component of the phenomenon (if \p in_cascade_estimation is true, contains only 0s).
#'                               \item 'A__': **`list of numeric matrices`**, containing the operator A_e for each statistical unit (if \p in_cascade_estimation is true, each matrix contains only 0s).
#'                               \item 'B__for_K': **`list of numeric matrices`**, containing the operator B_e used for the K_e_s(t) computation, for each statistical unit (if \p in_cascade_estimation is true, each matrix contains only 0s).
#'                               }
#'                         \item 'inputs_info': **`list`** containing information about the data used to fit the model:
#'                               \itemize{
#'                               \item 'Response': **`list`**:
#'                                     \itemize{
#'                                     \item 'basis_num': **`positive integer`**: number of basis used to make the basis expansion of the functional response (element \p n_basis_y_points).
#'                                     \item 'basis_type': **`string`**: basis used to make the basis expansion of the functional response. Possible values: **`"bsplines"`**, **`"constant"`** (element \p basis_type_y_points).
#'                                     \item 'basis_deg': **`positive integer`**: degree of basis used to make the basis expansion of the functional response (element \p degree_basis_y_points).
#'                                     \item 'knots': **`numeric vector`**: knots used to make the basis expansion of the functional response (element \p knots_y_points).
#'                                     \item 'basis_coeff': **`numeric matrix`**: coefficients of the basis expansion of the functional response (element \p coeff_y_points).
#'                                     }
#'                               \item 'ResponseReconstructionWeights': **`list`**:
#'                                     \itemize{
#'                                     \item 'basis_num': **`positive integer`**: number of basis used to make the basis expansion of the functional response (element \p n_basis_rec_weights_y_points).
#'                                     \item 'basis_type': **`string`**: basis used to make the basis expansion of the functional response. Possible values: **`"bsplines"`**, **`"constant"`** (element \p basis_type_rec_weights_y_points).
#'                                     \item 'basis_deg': **`positive integer`**: degree of basis used to make the basis expansion of the functional response (element \p degree_basis_rec_weights_y_points).
#'                                     \item 'knots': **`numeric vector`**: knots used to make the basis expansion of the functional response (element \p knots_y_points).
#'                                     \item 'basis_coeff': **`numeric matrix`**: coefficients of the basis expansion of the functional response (element \p coeff_rec_weights_y_points).
#'                                     } 
#'                               \item 'cov_Stationary': **`list`**:
#'                                     \itemize{
#'                                     \item 'number_covariates': **`positive integer`**: number of stationary covariates (length of \p coeff_stationary_cov).
#'                                     \item 'basis_num': **`integer vector`** of positive integer: numbers of basis used to make the basis expansion of the functional stationary covariates (respective elements of \p n_basis_stationary_cov).
#'                                     \item 'basis_type': **`vector of strings`**: types of basis used to make the basis expansion of the functional stationary covariates. Possible values: **`"bsplines"`**, **`"constant"`** (respective elements of \p basis_types_stationary_cov).
#'                                     \item 'basis_deg': **`integer vector`** of positive integers: degrees of basis used to make the basis expansion of the functional stationary covariates (respective elements of \p degrees_basis_stationary_cov).
#'                                     \item 'knots': **`numeric vector`**: knots used to make the basis expansion of the functional stationary covariates (respective elements of \p knots_stationary_cov).
#'                                     \item 'basis_coeff': **`list of numeric matrices`**: coefficients of the basis expansion of the functional stationary covariates  (respective elements of \p coeff_stationary_cov).
#'                                     }      
#'                               \item 'beta_Stationary': **`list`**:
#'                                     \itemize{
#'                                     \item 'basis_num': **`integer vector`** of positive integer: numbers of basis used to make the basis expansion of the functional regression coefficients of the stationary covariates (respective elements of \p n_basis_beta_stationary_cov).
#'                                     \item 'basis_type': **`vector of strings`**: types of basis used to make the basis expansion of the functional regression coefficients of the stationary covariates. Possible values: **`"bsplines"`**, **`"constant"`** (respective elements of \p basis_types_beta_stationary_cov).
#'                                     \item 'basis_deg': **`integer vector`** of positive integers: degrees of basis used to make the basis expansion of the functional regression coefficients of the stationary covariates (respective elements \p degrees_basis_beta_stationary_cov).
#'                                     \item 'knots': knots used to make the basis expansion of the functional regression coefficients of the stationary covariates (element \p knots_beta_stationary_cov).
#'                                     }
#'                               \item 'cov_Event': **`list`**:
#'                                     \itemize{
#'                                     \item 'number_covariates': **`positive integer`**: number of event-dependent covariates (length of \p coeff_events_cov).
#'                                     \item 'basis_num': **`integer vector`** of positive integer: numbers of basis used to make the basis expansion of the functional event-dependent covariates (respective elements of \p n_basis_events_cov).
#'                                     \item 'basis_type': **`vector of strings`**: types of basis used to make the basis expansion of the functional event-dependent covariates. Possible values: **`"bsplines"`**, **`"constant"`** (respective elements of \p basis_types_events_cov).
#'                                     \item 'basis_deg': **`integer vector`** of positive integers: degrees of basis used to make the basis expansion of the functional event-dependent covariates (respective elements \p degrees_basis_events_cov).
#'                                     \item 'knots': **`numeric vector`**: knots used to make the basis expansion of the functional event-dependent covariates (respective elements of \p knots_events_cov).
#'                                     \item 'basis_coeff': **`list of numeric matrices`**: coefficients of the basis expansion of the functional event-dependent covariates  (respective elements of \p coeff_events_cov).
#'                                     \item 'penalizations': **`numeric vector`** of positive double: penalizations of the event-dependent covariates (respective elements of \p penalization_events_cov)
#'                                     \item 'coordinates': **`numeric matrix`**: UTM coordinates of the events of the training data (element \p coordinates_events).
#'                                     \item 'kernel_bwd': **`double`**: bandwith of the gaussian kernel used to smooth distances of the events (element \p kernel_bandwith_events).
#'                                     }      
#'                               \item 'beta_Event': **`list`**:
#'                                     \itemize{
#'                                     \item 'basis_num': **`integer vector`** of positive integer: numbers of basis used to make the basis expansion of the functional regression coefficients of the event-dependent covariates (respective elements of \p n_basis_beta_events_cov).
#'                                     \item 'basis_type': **`vector of strings`**: types of basis used to make the basis expansion of the functional regression coefficients of the event-dependent covariates. Possible values: **`"bsplines"`**, **`"constant"`** (respective elements of \p basis_types_beta_events_cov).
#'                                     \item 'basis_deg': **`integer vector`** of positive integers: degrees of basis used to make the basis expansion of the functional regression coefficients of the event-dependent covariates (element \p degrees_basis_beta_events_cov).
#'                                     \item 'knots': knots used to make the basis expansion of the functional regression coefficients of the event-dependent covariates (element \p knots_beta_events_cov).
#'                                     }
#'                               \item 'cov_Station': **`list`**:
#'                                     \itemize{
#'                                     \item 'number_covariates': **`positive integer`**: number of station-dependent covariates (length of \p coeff_stations_cov).
#'                                     \item 'basis_num': **`integer vector`** of positive integer: numbers of basis used to make the basis expansion of the functional station-dependent covariates (respective elements of \p n_basis_stations_cov).
#'                                     \item 'basis_type': **`vector of strings`**: types of basis used to make the basis expansion of the functional station-dependent covariates. Possible values: **`"bsplines"`**, **`"constant"`** (respective elements of \p basis_types_stations_cov).
#'                                     \item 'basis_deg': **`integer vector`** of positive integers: degrees of basis used to make the basis expansion of the functional station-dependent covariates (respective elements \p degrees_basis_stations_cov).
#'                                     \item 'knots': **`numeric vector`**: knots used to make the basis expansion of the functional station-dependent covariates (respective elements of \p knots_stations_cov).
#'                                     \item 'basis_coeff': **`list of numeric matrices`**: coefficients of the basis expansion of the functional station-dependent covariates  (respective elements of \p coeff_stations_cov).
#'                                     \item 'penalizations': **`numeric vector`** of positive double: penalizations of the station-dependent covariates (respective elements of \p penalization_stations_cov)
#'                                     \item 'coordinates': **`numeric matrix`**: UTM coordinates of the stations of the training data (element \p coordinates_stations).
#'                                     \item 'kernel_bwd': **`double`**: bandwith of the gaussian kernel used to smooth distances of the stations (element \p kernel_bandwith_stations).
#'                                     }      
#'                               \item 'beta_Station': **`list`**:
#'                                     \itemize{
#'                                     \item 'basis_num': **`integer vector`** of positive integer: numbers of basis used to make the basis expansion of the functional regression coefficients of the station-dependent covariates (respective elements of \p n_basis_beta_stations_cov).
#'                                     \item 'basis_type': **`vector of strings`**: types of basis used to make the basis expansion of the functional regression coefficients of the station-dependent covariates. Possible values: **`"bsplines"`**, **`"constant"`** (respective elements of \p basis_types_beta_stations_cov).
#'                                     \item 'basis_deg': **`integer vector`** of positive integers: degrees of basis used to make the basis expansion of the functional regression coefficients of the station-dependent covariates (element \p degrees_basis_beta_stations_cov).
#'                                     \item 'knots': knots used to make the basis expansion of the functional regression coefficients of the station-dependent covariates (element \p knots_beta_stations_cov).
#'                                     }
#'                               }
#'                               \item 'a': **`double`**: domain left extreme  (element \p left_extreme_domain).
#'                               \item 'b': **`double`**: domain right extreme  (element \p right_extreme_domain).
#'                               \item 'abscissa': **`numeric vector`** of double: abscissa for which the evaluations of the functional data are available (element \p t_points).
#'                               \item 'InCascadeEstimation': element \p in_cascade_estimation.
#'                         }
#'                   }
#' @details
#' Constant basis are used, for a covariate, if it resembles a scalar shape. It consists of a straight line with y-value equal to 1 all over the data domain.
#' Can be seen as a B-spline basis with degree 0, number of basis 1, using one knot, consequently having only one coefficient for the only basis for each statistical unit.
#' fdagwr sets all the feats accordingly if reads constant basis.
#' However, recall that the response is a functional datum, as the regressors coefficients. Since the package's basis variety could be hopefully enlarged in the future 
#' (for example, introducing Fourier basis for handling data that present periodical behaviors), the input parameters regarding basis types for response, response reconstruction
#' weights and regressors coefficients are left at the end of the input list, and defaulted as **`NULL`**. Consequently they will use a B-spline basis system, and should NOT use a constant basis,
#' Recall to perform externally the basis expansion before using the package, and afterwards passing basis types, degree and number and basis expansion coefficients and knots coherently
#' @note 
#' A little excursion about degree and number of basis passed as input. For each specific covariate, or the response, if using B-spline basis, remember that number of knots = number of basis - degree + 1. 
#' By default, if passing **`NULL`**, fdagwr uses a cubic B-spline system of basis, the number of basis is computed coherently from the number of knots (that is the only mandatory input parameter).
#' Passing only the degree of the bsplines, the number of basis used will be set accordingly, and viceversa if passing only the number of basis. 
#' But, take care that the number of basis used has to match the number of rows of coefficients matrix (for EACH type of basis). If not, an exception is thrown. No problems arise if letting fdagwr defaulting the number of basis.
#' For response and response reconstruction weights, degree and number of basis consist of integer, and can be **`NULL`** For all the regressors, and their coefficients, the inputs consist of vector of integers: 
#' if willing to pass a default parameter, all the vector has to be defaulted (if passing **`NULL`**, a vector with all 3 for the degrees is passed, for example)
#' @references
#' - Paper: \href{https://www.researchgate.net/publication/377251714_Weighted_Functional_Data_Analysis_for_the_Calibration_of_a_Ground_Motion_Model_in_Italy}{Functional Response Reconstruction Weights}
#' - Source code: \href{https://github.com/AndreaEnricoFranzoni/FunctionalMultiSourceGeographicallyWeightedRegression}{fdagwr implementation}
#' @export
#' @author Andrea Enrico Franzoni
NULL



#' @title predict_FMSGWR_ESC
#' @name predict_FMSGWR_ESC
#' @description
#' Function to perform predictions on new statistical units using a fitted Functional Multi-Source Geographically Weighted Regression ESC model. Non-stationary betas have to be recomputed in the new locations.
#' @param coeff_stationary_cov_to_pred **`list of numeric matrices`** of double: element i-th containing the coefficients for the basis expansion of the i-th stationary covariate to be predicted: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit to be predicted.
#' @param coeff_events_cov_to_pred **`list of numeric matrices`** of double: element i-th containing the coefficients for the basis expansion of the i-th event-dependent covariate to be predicted: each row represents a specific basis (by default: B-spline) of the basis system used, each column a statistical unit to be predicted.
#' @param coordinates_events_to_pred **`numeric matrix`**  of double containing the UTM coordinates of the event of new statistical units: each row represents a statistical unit to be predicted, each column a coordinate (2 columns).
#' @param coeff_stations_cov_to_pred **`list of numeric matrices`** of double: element i-th containing the coefficients for the basis expansion of the i-th station-dependent covariate to be predicted: each row represents a specific basis (by default: B-spline) of the basis system used, each column a new statistical unit.
#' @param coordinates_stations_to_pred **`numeric matrix`** of double containing the UTM coordinates of the station of new statistical units: each row represents a statistical unit to be predicted, each column a coordinate (2 columns).
#' @param abscissa_ev **`numeric vector`** of double: abscissa for which then evaluating the predicted response and betas, stationary and non-stationary, which have to be recomputed
#' @param model_fitted **`list`** containing:
#'                   \itemize{
#'                   \item 'FGWR': **`string`**: the type of fgwr used (**`"FMSGWR_ESC"`**);
#'                   \item 'EstimationTechnique': **`string`**: **`"Exact"`** if \p in_cascade_estimation false, **`"Cascade"`** if \p in_cascade_estimation true 
#'                   \item 'Bc': **`list`** containing, for each stationary covariate regression coefficient (each element is named with the element names in the list \p coeff_stationary_cov (default, if not given: **`"CovC*"`**)):
#'                         \itemize{
#'                         \item 'basis_coeff': **`numeric vector`** of double, dimension Lc_jx1,containing the coefficients of the basis expansion of the beta.
#'                         \item 'basis_type': **`string`** containing the basis type over which the beta basis expansion is performed. Possible values: **`"bsplines"`**, **`"constant"`**. (Respective element of \p basis_types_beta_stationary_cov).
#'                         \item 'basis_num': **`positive integer`**: the number of basis used for performing the beta basis expansion (respective elements of \p n_basis_beta_stationary_cov).
#'                         \item 'knots': **`numeric vector`** of double: knots used to create the basis system for the beta (it is the input \p knots_beta_stationary_cov).
#'                         }
#'                   \item 'Beta_c': **`list`** containing, for each stationary covariate regression coefficient (each element is named with the element names in the list coeff_stationary_cov (default, if not given: **`"CovC*"`**)) a list with:
#'                         \itemize{
#'                         \item 'Beta_eval': **`numeric vector`** of double containing the discrete evaluations of the stationary beta.
#'                         \item 'Abscissa': **`numeric vector`** of double containing the domain points for which the evaluation of the beta is available (it is the input \p t_points).
#'                         }
#'                   \item 'Be': **`list`** containing, for each event-dependent covariate regression coefficient (each element is named with the element names in the list \p coeff_events_cov (default, if not given: **`"CovE*"`**)):
#'                         \itemize{
#'                         \item 'basis_coeff': **`list`**, containing, for each statistical unit, a **`numeric vector`** of double, dimension Le_jx1,containing the coefficients of the basis expansion of the beta for that unit.
#'                         \item 'basis_type': **`string`** containing the basis type over which the beta basis expansion is performed. Possible values: **`"bsplines"`**, **`"constant"`**. (Respective element of \p basis_types_beta_events_cov).
#'                         \item 'basis_num': **`positive integer`**: the number of basis used for performing the beta basis expansion (respective elements of \p n_basis_beta_events_cov).
#'                         \item 'knots': **`numeric vector`** of double: knots used to create the basis system for the beta (it is the input \p knots_beta_events_cov).
#'                         }
#'                   \item 'Beta_e': **`list`** containing, for each event-dependent covariate regression coefficient (each element is named with the element names in the list \p coeff_events_cov (default, if not given: **`"CovE*"`**)) a list with:
#'                         \itemize{
#'                         \item 'Beta_eval': **`list`**, containing, for each statistical unit, a **`numeric vector`** of double containing the discrete evaluations of the non-stationary beta, one for each unit.
#'                         \item 'Abscissa': **`numeric vector`** of double containing the domain points for which the evaluation of the beta is available (it is the input \p t_points).
#'                         }
#'                   \item 'Bs': **`list`** containing, for each station-dependent covariate regression coefficient (each element is named with the element names in the list \p coeff_stations_cov (default, if not given: **`"CovS*"`**)):
#'                         \itemize{
#'                         \item 'basis_coeff': **`list`**, containing, for each statistical unit, a **`numeric vector`** of double, dimension Ls_jx1,containing the coefficients of the basis expansion of the beta for that unit.
#'                         \item 'basis_type': **`string`** containing the basis type over which the beta basis expansion is performed. Possible values: **`"bsplines"`**, **`"constant"`**. (Respective element of \p basis_types_beta_stations_cov).
#'                         \item 'basis_num': **`positive integer`**: the number of basis used for performing the beta basis expansion (respective elements of \p n_basis_beta_stations_cov).
#'                         \item 'knots': **`numeric vector`** of double: knots used to create the basis system for the beta (it is the input \p knots_beta_stations_cov).
#'                         }
#'                   \item 'Beta_e': **`list`** containing, for each station-dependent covariate regression coefficient (each element is named with the element names in the list \p coeff_stations_cov (default, if not given: **`"CovS*"`**)) a list with:
#'                         \itemize{
#'                         \item 'Beta_eval': **`list`**, containing, for each statistical unit, a **`numeric vector`** of double containing the discrete evaluations of the non-stationary beta, one for each unit.
#'                         \item 'Abscissa': **`numeric vector`** of double containing the domain points for which the evaluation of the beta is available (it is the input \p t_points).
#'                         }
#'                   \item 'predictor_info': **`list`** containing partial residuals and information of the fitted model to perform predictions for new statistical units:
#'                         \itemize{
#'                         \item 'partial_res': **`list`** containing information to compute the partial residuals:
#'                               \itemize{
#'                               \item 'c_tilde_hat': **`numeric vector`** of double with the basis expansion coefficients of the response minus the stationary component of the phenomenon (if \p in_cascade_estimation is true, contains only 0s).
#'                               \item 'A__': **`list of numeric matrices`**, containing the operator A_e for each statistical unit (if \p in_cascade_estimation is true, each matrix contains only 0s).
#'                               \item 'B__for_K': **`list of numeric matrices`**, containing the operator B_e used for the K_e_s(t) computation, for each statistical unit (if \p in_cascade_estimation is true, each matrix contains only 0s).
#'                               }
#'                         \item 'inputs_info': **`list`** containing information about the data used to fit the model:
#'                               \itemize{
#'                               \item 'Response': **`list`**:
#'                                     \itemize{
#'                                     \item 'basis_num': **`positive integer`**: number of basis used to make the basis expansion of the functional response (element \p n_basis_y_points).
#'                                     \item 'basis_type': **`string`**: basis used to make the basis expansion of the functional response. Possible values: **`"bsplines"`**, **`"constant"`** (element \p basis_type_y_points).
#'                                     \item 'basis_deg': **`positive integer`**: degree of basis used to make the basis expansion of the functional response (element \p degree_basis_y_points).
#'                                     \item 'knots': **`numeric vector`**: knots used to make the basis expansion of the functional response (element \p knots_y_points).
#'                                     \item 'basis_coeff': **`numeric matrix`**: coefficients of the basis expansion of the functional response (element \p coeff_y_points).
#'                                     }
#'                               \item 'ResponseReconstructionWeights': **`list`**:
#'                                     \itemize{
#'                                     \item 'basis_num': **`positive integer`**: number of basis used to make the basis expansion of the functional response (element \p n_basis_rec_weights_y_points).
#'                                     \item 'basis_type': **`string`**: basis used to make the basis expansion of the functional response. Possible values: **`"bsplines"`**, **`"constant"`** (element \p basis_type_rec_weights_y_points).
#'                                     \item 'basis_deg': **`positive integer`**: degree of basis used to make the basis expansion of the functional response (element \p degree_basis_rec_weights_y_points).
#'                                     \item 'knots': **`numeric vector`**: knots used to make the basis expansion of the functional response (element \p knots_y_points).
#'                                     \item 'basis_coeff': **`numeric matrix`**: coefficients of the basis expansion of the functional response (element \p coeff_rec_weights_y_points).
#'                                     } 
#'                               \item 'cov_Stationary': **`list`**:
#'                                     \itemize{
#'                                     \item 'number_covariates': **`positive integer`**: number of stationary covariates (length of \p coeff_stationary_cov).
#'                                     \item 'basis_num': **`integer vector`** of positive integer: numbers of basis used to make the basis expansion of the functional stationary covariates (respective elements of \p n_basis_stationary_cov).
#'                                     \item 'basis_type': **`vector of strings`**: types of basis used to make the basis expansion of the functional stationary covariates. Possible values: **`"bsplines"`**, **`"constant"`** (respective elements of \p basis_types_stationary_cov).
#'                                     \item 'basis_deg': **`integer vector`** of positive integers: degrees of basis used to make the basis expansion of the functional stationary covariates (respective elements of \p degrees_basis_stationary_cov).
#'                                     \item 'knots': **`numeric vector`**: knots used to make the basis expansion of the functional stationary covariates (respective elements of \p knots_stationary_cov).
#'                                     \item 'basis_coeff': **`list of numeric matrices`**: coefficients of the basis expansion of the functional stationary covariates  (respective elements of \p coeff_stationary_cov).
#'                                     }      
#'                               \item 'beta_Stationary': **`list`**:
#'                                     \itemize{
#'                                     \item 'basis_num': **`integer vector`** of positive integer: numbers of basis used to make the basis expansion of the functional regression coefficients of the stationary covariates (respective elements of \p n_basis_beta_stationary_cov).
#'                                     \item 'basis_type': **`vector of strings`**: types of basis used to make the basis expansion of the functional regression coefficients of the stationary covariates. Possible values: **`"bsplines"`**, **`"constant"`** (respective elements of \p basis_types_beta_stationary_cov).
#'                                     \item 'basis_deg': **`integer vector`** of positive integers: degrees of basis used to make the basis expansion of the functional regression coefficients of the stationary covariates (respective elements \p degrees_basis_beta_stationary_cov).
#'                                     \item 'knots': knots used to make the basis expansion of the functional regression coefficients of the stationary covariates (element \p knots_beta_stationary_cov).
#'                                     }
#'                               \item 'cov_Event': **`list`**:
#'                                     \itemize{
#'                                     \item 'number_covariates': **`positive integer`**: number of event-dependent covariates (length of \p coeff_events_cov).
#'                                     \item 'basis_num': **`integer vector`** of positive integer: numbers of basis used to make the basis expansion of the functional event-dependent covariates (respective elements of \p n_basis_events_cov).
#'                                     \item 'basis_type': **`vector of strings`**: types of basis used to make the basis expansion of the functional event-dependent covariates. Possible values: **`"bsplines"`**, **`"constant"`** (respective elements of \p basis_types_events_cov).
#'                                     \item 'basis_deg': **`integer vector`** of positive integers: degrees of basis used to make the basis expansion of the functional event-dependent covariates (respective elements \p degrees_basis_events_cov).
#'                                     \item 'knots': **`numeric vector`**: knots used to make the basis expansion of the functional event-dependent covariates (respective elements of \p knots_events_cov).
#'                                     \item 'basis_coeff': **`list of numeric matrices`**: coefficients of the basis expansion of the functional event-dependent covariates  (respective elements of \p coeff_events_cov).
#'                                     \item 'penalizations': **`numeric vector`** of positive double: penalizations of the event-dependent covariates (respective elements of \p penalization_events_cov)
#'                                     \item 'coordinates': **`numeric matrix`**: UTM coordinates of the events of the training data (element \p coordinates_events).
#'                                     \item 'kernel_bwd': **`double`**: bandwith of the gaussian kernel used to smooth distances of the events (element \p kernel_bandwith_events).
#'                                     }      
#'                               \item 'beta_Event': **`list`**:
#'                                     \itemize{
#'                                     \item 'basis_num': **`integer vector`** of positive integer: numbers of basis used to make the basis expansion of the functional regression coefficients of the event-dependent covariates (respective elements of \p n_basis_beta_events_cov).
#'                                     \item 'basis_type': **`vector of strings`**: types of basis used to make the basis expansion of the functional regression coefficients of the event-dependent covariates. Possible values: **`"bsplines"`**, **`"constant"`** (respective elements of \p basis_types_beta_events_cov).
#'                                     \item 'basis_deg': **`integer vector`** of positive integers: degrees of basis used to make the basis expansion of the functional regression coefficients of the event-dependent covariates (element \p degrees_basis_beta_events_cov).
#'                                     \item 'knots': knots used to make the basis expansion of the functional regression coefficients of the event-dependent covariates (element \p knots_beta_events_cov).
#'                                     }
#'                               \item 'cov_Station': **`list`**:
#'                                     \itemize{
#'                                     \item 'number_covariates': **`positive integer`**: number of station-dependent covariates (length of \p coeff_stations_cov).
#'                                     \item 'basis_num': **`integer vector`** of positive integer: numbers of basis used to make the basis expansion of the functional station-dependent covariates (respective elements of \p n_basis_stations_cov).
#'                                     \item 'basis_type': **`vector of strings`**: types of basis used to make the basis expansion of the functional station-dependent covariates. Possible values: **`"bsplines"`**, **`"constant"`** (respective elements of \p basis_types_stations_cov).
#'                                     \item 'basis_deg': **`integer vector`** of positive integers: degrees of basis used to make the basis expansion of the functional station-dependent covariates (respective elements \p degrees_basis_stations_cov).
#'                                     \item 'knots': **`numeric vector`**: knots used to make the basis expansion of the functional station-dependent covariates (respective elements of \p knots_stations_cov).
#'                                     \item 'basis_coeff': **`list of numeric matrices`**: coefficients of the basis expansion of the functional station-dependent covariates  (respective elements of \p coeff_stations_cov).
#'                                     \item 'penalizations': **`numeric vector`** of positive double: penalizations of the station-dependent covariates (respective elements of \p penalization_stations_cov)
#'                                     \item 'coordinates': **`numeric matrix`**: UTM coordinates of the stations of the training data (element \p coordinates_stations).
#'                                     \item 'kernel_bwd': **`double`**: bandwith of the gaussian kernel used to smooth distances of the stations (element \p kernel_bandwith_stations).
#'                                     }      
#'                               \item 'beta_Station': **`list`**:
#'                                     \itemize{
#'                                     \item 'basis_num': **`integer vector`** of positive integer: numbers of basis used to make the basis expansion of the functional regression coefficients of the station-dependent covariates (respective elements of \p n_basis_beta_stations_cov).
#'                                     \item 'basis_type': **`vector of strings`**: types of basis used to make the basis expansion of the functional regression coefficients of the station-dependent covariates. Possible values: **`"bsplines"`**, **`"constant"`** (respective elements of \p basis_types_beta_stations_cov).
#'                                     \item 'basis_deg': **`integer vector`** of positive integers: degrees of basis used to make the basis expansion of the functional regression coefficients of the station-dependent covariates (element \p degrees_basis_beta_stations_cov).
#'                                     \item 'knots': knots used to make the basis expansion of the functional regression coefficients of the station-dependent covariates (element \p knots_beta_stations_cov).
#'                                     }
#'                               }
#'                               \item 'a': **`double`**: domain left extreme  (element \p left_extreme_domain).
#'                               \item 'b': **`double`**: domain right extreme  (element \p right_extreme_domain).
#'                               \item 'abscissa': **`numeric vector`** of double: abscissa for which the evaluations of the functional data are available (element \p t_points).
#'                               \item 'InCascadeEstimation': element \p in_cascade_estimation.
#'                         }
#'                   }
#' @param n_knots_smoothing_pred **`positive integer`**: number of knots used to smooth predicted response and non-stationary, obtaining basis expansion coefficients with respect to the training basis (default: **`100`**).
#' @param n_intervals_quadrature **`positive integer`**: number of intervals used while performing integration via midpoint (rectangles) quadrature rule (default: **`100`**).
#' @param num_threads **`positive integer`**: number of threads to be used in OMP parallel directives. Default: **`NULL`**, that is equivalent to the maximum number of cores available in the machine.
#' @return **`list`** containing:
#'         \itemize{
#'         \item 'FGWR_predictor': **`string`**: model used to predict (**`"predictor_FMSGWR_ESC"`**).
#'         \item 'EstimationTechnique': **`string`**: **`"Exact"`** if *in_cascade_estimation* \p model_fitted false, **`"Cascade"`** if *in_cascade_estimation* \p model_fitted true.
#'         \item 'prediction': **`list`** containing:
#'               \itemize{
#'               \item 'evaluation': **`list`** containing the evaluation of the prediction:
#'                     \itemize{
#'                     \item 'prediction_ev': **`list`** containing, for each unit to be predicted, the raw evaluations of the predicted response.
#'                     \item 'abscissa_ev': **`numeric vector`** containing the abscissa points for which the prediction evaluation is available (element \p abscissa_ev).
#'                     }
#'               \item 'fd': **`list`** containing the prediction functional description:
#'                     \itemize{
#'                     \item 'prediction_basis_coeff': **`numeric matrix`** containing the prediction basis expansion coefficients (each row a basis, each column a new statistical unit).
#'                     \item 'prediction_basis_type': **`string`**: basis type used for the predicted response basis expansion (from \p model_fitted).
#'                     \item 'prediction_basis_num': **`positive integer`**: number of basis used for the predicted response basis expansion (from \p model_fitted).
#'                     \item 'prediction_basis_deg': **`positive integer`**: degree of basis used for the predicted response basis expansion (from \p model_fitted)
#'                     \item 'prediction_knots': **`numeric vector`**: knots used for the predicted response smoothing (\p n_knots_smoothing_pred equally spaced knots in the functional datum domain)
#'                     }
#'               }
#'         \item 'Bc_pred': **`list`** containing, for each stationary covariate:
#'               \itemize{
#'               \item 'basis_coeff': **`numeric matrix`** of double containing the fitted basis expansion coefficients of the beta (from \p model_fitted).
#'               \item 'basis_num': **`positive integer`**: number of basis used for the beta basis expansion (from \p model_fitted).
#'               \item 'basis_type': type of basis used for the beta basis expansion (from \p model_fitted).
#'               \item 'knots': knots used for the beta basis expansion (from \p model_fitted).
#'               }
#'         \item 'Beta_c_pred': **`list`** containing, for each stationary covariate:
#'               \itemize{
#'               \item 'Beta_eval': **`numeric vector`** of double containing the evaluation of the beta along a grid.
#'               \item 'basis_coeff': **`numeric vector`** of double containing the grid (element \p abscissa_ev).
#'               }
#'         \item 'Be_pred': **`list`** containing, for each event-dependent covariate:
#'               \itemize{
#'               \item 'basis_coeff': **`list of numeric matrix`** of double: one element for each unit to be predicted containing the recomputed basis expansion coefficients of the beta on the locations of the predicted units.
#'               \item 'basis_num': **`positive integer`**: number of basis used for the beta basis expansion (from \p model_fitted).
#'               \item 'basis_type': type of basis used for the beta basis expansion (from \p model_fitted).
#'               \item 'knots': knots used for the beta basis expansion (from \p model_fitted).
#'               }
#'         \item 'Beta_e_pred': **`list`** containing, for each event-dependent covariate:
#'               \itemize{
#'               \item 'Beta_eval': **`list of numeric vector`** of double containing, for each unit to be predicted, the evaluation of the beta along a grid.
#'               \item 'basis_coeff': **`numeric vector`** of double containing the grid (element \p abscissa_ev).
#'               } 
#'         \item 'Bs_pred': **`list`** containing, for each station-dependent covariate:
#'               \itemize{
#'               \item 'basis_coeff': **`list of numeric matrix`** of double: one element for each unit to be predicted containing the recomputed basis expansion coefficients of the beta on the locations of the predicted units.
#'               \item 'basis_num': **`positive integer`**: number of basis used for the beta basis expansion (from \p model_fitted).
#'               \item 'basis_type': type of basis used for the beta basis expansion (from \p model_fitted).
#'               \item 'knots': knots used for the beta basis expansion (from \p model_fitted).
#'               }
#'         \item 'Beta_s_pred': **`list`** containing, for each station-dependent covariate:
#'               \itemize{
#'               \item 'Beta_eval': **`list of numeric vector`** of double containing, for each unit to be predicted, the evaluation of the beta along a grid.
#'               \item 'basis_coeff': **`numeric vector`** of double containing the grid (element \p abscissa_ev).
#'               }         
#'         }
#' @details
#' Covariates of units to be predicted have to be sampled in the same sample points for which the training data have been (t_points of FMSGWR_ESC).
#' Covariates basis expansion for the units to be predicted has to be done with respect to the basis used for the covariates in the training set
#' @references
#' - Source code: \href{https://github.com/AndreaEnricoFranzoni/FunctionalMultiSourceGeographicallyWeightedRegression}{fdagwr implementation}
#' @export
#' @author Andrea Enrico Franzoni
NULL
