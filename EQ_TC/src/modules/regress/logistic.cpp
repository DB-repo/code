/* ------------------------------------------------------
 *
 * @file logistic.cpp
 *
 * @brief Logistic-Regression functions
 *
 * We implement the conjugate-gradient method and the iteratively-reweighted-
 * least-squares method.
 *
 *//* ----------------------------------------------------------------------- */
#include <limits>
#include <dbconnector/dbconnector.hpp>
#include <modules/shared/HandleTraits.hpp>
#include <modules/prob/boost.hpp>
#include <boost/math/distributions.hpp>
#include <modules/prob/student.hpp>
#include "logistic.hpp"

namespace madlib {

// Use Eigen
using namespace dbal::eigen_integration;

namespace modules {

// Import names from other MADlib modules
using dbal::NoSolutionFoundException;

namespace regress {

// FIXME this enum should be accessed by all modules that may need grouping
// valid status values
enum { IN_PROCESS, COMPLETED, TERMINATED, NULL_EMPTY };

// Internal functions
AnyType stateToResult(const Allocator &inAllocator,
                      const HandleMap<const ColumnVector, TransparentHandle<double> >& inCoef,
                      const Matrix & hessian,
                      const double &logLikelihood,
                      int status,
                      const uint64_t &numRows);


// ---------------------------------------------------------------------------
//              Logistic Regression States
// ---------------------------------------------------------------------------

/**
 * @brief Inter- and intra-iteration state for conjugate-gradient method for
 *        logistic regression
 *
 * TransitionState encapsualtes the transition state during the
 * logistic-regression aggregate function. To the database, the state is
 * exposed as a single DOUBLE PRECISION array, to the C++ code it is a proper
 * object containing scalars and vectors.
 *
 * Note: We assume that the DOUBLE PRECISION array is initialized by the
 * database with length at least 5, and all elemenets are 0.
 *
 */
template <class Handle>
class LogRegrCGTransitionState {
    template <class OtherHandle>
    friend class LogRegrCGTransitionState;

  public:
    LogRegrCGTransitionState(const AnyType &inArray)
        : mStorage(inArray.getAs<Handle>()) {

        rebind(static_cast<uint16_t>(mStorage[1]));
    }

    /**
     * @brief Convert to backend representation
     *
     * We define this function so that we can use State in the
     * argument list and as a return type.
     */
    inline operator AnyType() const {
        return mStorage;
    }

    /**
     * @brief Initialize the conjugate-gradient state.
     *
     * This function is only called for the first iteration, for the first row.
     */
    inline void initialize(const Allocator &inAllocator, uint16_t inWidthOfX) {
        mStorage = inAllocator.allocateArray<double, dbal::AggregateContext,
                                             dbal::DoZero, dbal::ThrowBadAlloc>(arraySize(inWidthOfX));
        rebind(inWidthOfX);
        widthOfX = inWidthOfX;
    }

    /**
     * @brief We need to support assigning the previous state
     */
    template <class OtherHandle>
    LogRegrCGTransitionState &operator=(
        const LogRegrCGTransitionState<OtherHandle> &inOtherState) {

        for (size_t i = 0; i < mStorage.size(); i++)
            mStorage[i] = inOtherState.mStorage[i];
        return *this;
    }

    /**
     * @brief Merge with another State object by copying the intra-iteration
     *     fields
     */
    template <class OtherHandle>
    LogRegrCGTransitionState &operator+=(
        const LogRegrCGTransitionState<OtherHandle> &inOtherState) {

        if (mStorage.size() != inOtherState.mStorage.size() ||
            widthOfX != inOtherState.widthOfX)
            throw std::logic_error("Internal error: Incompatible transition "
                                   "states");

        numRows += inOtherState.numRows;
        gradNew += inOtherState.gradNew;
        X_transp_AX += inOtherState.X_transp_AX;
        logLikelihood += inOtherState.logLikelihood;
        // merged state should have the higher status
        // (see top of file for more on 'status' )
        status = (inOtherState.status > status) ? inOtherState.status : status;
        return *this;
    }

    /**
     * @brief Reset the inter-iteration fields.
     */
    inline void reset() {
        numRows = 0;
        X_transp_AX.fill(0);
        gradNew.fill(0);
        logLikelihood = 0;
        status = IN_PROCESS;
    }

  private:
    static inline size_t arraySize(const uint16_t inWidthOfX) {
        return 6 + inWidthOfX * inWidthOfX + 4 * inWidthOfX;
    }

    /**
     * @brief Rebind to a new storage array
     *
     * @param inWidthOfX The number of independent variables.
     *
     * Array layout (iteration refers to one aggregate-function call):
     * Inter-iteration components (updated in final function):
     * - 0: iteration (current iteration)
     * - 1: widthOfX (number of coefficients)
     * - 2: coef (vector of coefficients)
     * - 2 + widthOfX: dir (direction)
     * - 2 + 2 * widthOfX: grad (gradient)
     * - 2 + 3 * widthOfX: beta (scale factor)
     *
     * Intra-iteration components (updated in transition step):
     * - 3 + 3 * widthOfX: numRows (number of rows already processed in this iteration)
     * - 4 + 3 * widthOfX: gradNew (intermediate value for gradient)
     * - 4 + 4 * widthOfX: X_transp_AX (X^T A X)
     * - 4 + widthOfX * widthOfX + 4 * widthOfX: logLikelihood ( ln(l(c)) )
     */
    void rebind(uint16_t inWidthOfX) {
        iteration.rebind(&mStorage[0]);
        widthOfX.rebind(&mStorage[1]);
        coef.rebind(&mStorage[2], inWidthOfX);
        dir.rebind(&mStorage[2 + inWidthOfX], inWidthOfX);
        grad.rebind(&mStorage[2 + 2 * inWidthOfX], inWidthOfX);
        beta.rebind(&mStorage[2 + 3 * inWidthOfX]);
        numRows.rebind(&mStorage[3 + 3 * inWidthOfX]);
        gradNew.rebind(&mStorage[4 + 3 * inWidthOfX], inWidthOfX);
        X_transp_AX.rebind(&mStorage[4 + 4 * inWidthOfX], inWidthOfX, inWidthOfX);
        logLikelihood.rebind(&mStorage[4 + inWidthOfX * inWidthOfX + 4 * inWidthOfX]);
        status.rebind(&mStorage[5 + inWidthOfX * inWidthOfX + 4 * inWidthOfX]);
    }

    Handle mStorage;

  public:
    typename HandleTraits<Handle>::ReferenceToUInt32 iteration;
    typename HandleTraits<Handle>::ReferenceToUInt16 widthOfX;
    typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap coef;
    typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap dir;
    typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap grad;
    typename HandleTraits<Handle>::ReferenceToDouble beta;

    typename HandleTraits<Handle>::ReferenceToUInt64 numRows;
    typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap gradNew;
    typename HandleTraits<Handle>::MatrixTransparentHandleMap X_transp_AX;
    typename HandleTraits<Handle>::ReferenceToDouble logLikelihood;
    typename HandleTraits<Handle>::ReferenceToUInt16 status;
};




/**
 * @brief Logistic function
 */
inline double sigma(double x) {
    return 1. / (1. + std::exp(-x));
}

/**
 * @brief Perform the logistic-regression transition step
 */
AnyType
logregr_cg_step_transition::run(AnyType &args) {
    LogRegrCGTransitionState<MutableArrayHandle<double> > state = args[0];
    if (args[1].isNull() || args[2].isNull()) { return args[0]; }
    double y = args[1].getAs<bool>() ? 1. : -1.;
    MappedColumnVector x;
    try {
        // an exception is raised in the backend if args[2] contains nulls
        MappedColumnVector xx = args[2].getAs<MappedColumnVector>();
        // x is a const reference, we can only rebind to change its pointer
        x.rebind(xx.memoryHandle(), xx.size());
    } catch (const ArrayWithNullException &e) {
        return args[0];
    }

    // The following check was added with MADLIB-138.
    if (!dbal::eigen_integration::isfinite(x)) {
        //throw std::domain_error("Design matrix is not finite.");
        warning("Design matrix is not finite.");
        state.status = TERMINATED;
        return state;
    }

    if (state.numRows == 0) {
        if (x.size() > std::numeric_limits<uint16_t>::max()){
            //throw std::domain_error(
            //    "Number of independent variables cannot be "
            //    "larger than 65535.");
            warning("Number of independent variables cannot be larger than 65535.");
            state.status = TERMINATED;
            return state;
        }


        state.initialize(*this, static_cast<uint16_t>(x.size()));
        if (!args[3].isNull()) {
            LogRegrCGTransitionState<ArrayHandle<double> > previousState = args[3];

            state = previousState;
            state.reset();
        }
    }
    // Now do the transition step
    state.numRows++;
    double xc = dot(x, state.coef);
    state.gradNew.noalias() += sigma(-y * xc) * y * trans(x);

    // Note: sigma(-x) = 1 - sigma(x).
    // a_i = sigma(x_i c) sigma(-x_i c)
    double a = sigma(xc) * sigma(-xc);
    //triangularView<Lower>(state.X_transp_AX) += x * trans(x) * a;
    state.X_transp_AX += x * trans(x) * a;

    //          n
    //         --
    // l(c) = -\  log(1 + exp(-y_i * c^T x_i))
    //         /_
    //         i=1
    state.logLikelihood -= std::log( 1. + std::exp(-y * xc) );

    return state;
}


/**
 * @brief Perform the perliminary aggregation function: Merge transition states
 */
AnyType
logregr_cg_step_merge_states::run(AnyType &args) {
    LogRegrCGTransitionState<MutableArrayHandle<double> > stateLeft = args[0];
    LogRegrCGTransitionState<ArrayHandle<double> > stateRight = args[1];

    // We first handle the trivial case where this function is called with one
    // of the states being the initial state
    if (stateLeft.numRows == 0)
        return stateRight;
    else if (stateRight.numRows == 0)
        return stateLeft;

    // Merge states together and return
    stateLeft += stateRight;
    return stateLeft;
}



/**
 * @brief Perform the logistic-regression final step
 */
AnyType
logregr_cg_step_final::run(AnyType &args) {
    // We request a mutable object. Depending on the backend, this might perform
    // a deep copy.
    LogRegrCGTransitionState<MutableArrayHandle<double> > state = args[0];

    // Aggregates that haven't seen any data just return Null.
    if (state.numRows == 0){
        state.status = NULL_EMPTY;
        return state;
    }

    // Note: k = state.iteration
    if (state.iteration == 0) {
        // Iteration computes the gradient

        state.dir = state.gradNew;
        state.grad = state.gradNew;
    } else {
        // We use the Hestenes-Stiefel update formula:
        //
        //            g_k^T (g_k - g_{k-1})
        // beta_k = -------------------------
        //          d_{k-1}^T (g_k - g_{k-1})
        ColumnVector gradNewMinusGrad = state.gradNew - state.grad;
        state.beta
            = dot(state.gradNew, gradNewMinusGrad)
            / dot(state.dir, gradNewMinusGrad);

        // Alternatively, we could use Polak-Ribi??re
        // state.beta
        //     = dot(state.gradNew, gradNewMinusGrad)
        //     / dot(state.grad, state.grad);

        // Or Fletcher???Reeves
        // state.beta
        //     = dot(state.gradNew, state.gradNew)
        //     / dot(state.grad, state.grad);

        // Do a direction restart (Powell restart)
        // Note: This is testing whether state.beta < 0 if state.beta were
        // assigned according to Polak-Ribi??re
        if (dot(state.gradNew, gradNewMinusGrad)
            / dot(state.grad, state.grad) <= std::numeric_limits<double>::denorm_min()) state.beta = 0;

        // d_k = g_k - beta_k * d_{k-1}
        state.dir = state.gradNew - state.beta * state.dir;
        state.grad = state.gradNew;
    }

    // H_k = - X^T A_k X
    // where A_k = diag(a_1, ..., a_n) and a_i = sigma(x_i c_{k-1}) sigma(-x_i c_{k-1})
    //
    //             g_k^T d_k
    // alpha_k = -------------
    //           d_k^T H_k d_k
    //
    // c_k = c_{k-1} - alpha_k * d_k
    state.coef += dot(state.grad, state.dir) /
        as_scalar(trans(state.dir) * state.X_transp_AX * state.dir)
        * state.dir;

    if(!state.coef.is_finite()){
        //throw NoSolutionFoundException(
        //    "Over- or underflow in conjugate-gradient step, while updating "
        //    "coefficients. Input data is likely of poor numerical condition.");
        warning("Over- or underflow in conjugate-gradient step, while updating "
              "coefficients. Input data is likely of poor numerical condition.");
        state.status = TERMINATED;
        return state;
    }

    state.iteration++;
    return state;
}

/**
 * @brief Return the difference in log-likelihood between two states
 */
AnyType
internal_logregr_cg_step_distance::run(AnyType &args) {
    LogRegrCGTransitionState<ArrayHandle<double> > stateLeft = args[0];
    LogRegrCGTransitionState<ArrayHandle<double> > stateRight = args[1];

    if(stateLeft.status == NULL_EMPTY || stateRight.status == NULL_EMPTY){
        return 0.0;
    }

    return std::abs(stateLeft.logLikelihood - stateRight.logLikelihood);
}

/**
 * @brief Return the coefficients and diagnostic statistics of the state
 */
AnyType
internal_logregr_cg_result::run(AnyType &args) {
    LogRegrCGTransitionState<ArrayHandle<double> > state = args[0];
    if (state.status == NULL_EMPTY)
        return Null();

    SymmetricPositiveDefiniteEigenDecomposition<Matrix> decomposition(
        state.X_transp_AX, EigenvaluesOnly, ComputePseudoInverse);

    return stateToResult(*this, state.coef,
                         state.X_transp_AX, state.logLikelihood,
                         state.status, state.numRows);
}



/**
 * @brief Inter- and intra-iteration state for iteratively-reweighted-least-
 *        squares method for logistic regression
 *
 * TransitionState encapsualtes the transition state during the
 * logistic-regression aggregate function. To the database, the state is
 * exposed as a single DOUBLE PRECISION array, to the C++ code it is a proper
 * object containing scalars, a vector, and a matrix.
 *
 * Note: We assume that the DOUBLE PRECISION array is initialized by the
 * database with length at least 4, and all elemenets are 0.
 */
template <class Handle>
class LogRegrIRLSTransitionState {
    template <class OtherHandle>
    friend class LogRegrIRLSTransitionState;

  public:
    LogRegrIRLSTransitionState(const AnyType &inArray)
        : mStorage(inArray.getAs<Handle>()) {

        rebind(static_cast<uint16_t>(mStorage[0]));
    }

    /**
     * @brief Convert to backend representation
     *
     * We define this function so that we can use State in the
     * argument list and as a return type.
     */
    inline operator AnyType() const {
        return mStorage;
    }

    /**
     * @brief Initialize the iteratively-reweighted-least-squares state.
     *
     * This function is only called for the first iteration, for the first row.
     */
    inline void initialize(const Allocator &inAllocator, uint16_t inWidthOfX) {
        mStorage = inAllocator.allocateArray<double, dbal::AggregateContext,
                                             dbal::DoZero, dbal::ThrowBadAlloc>(arraySize(inWidthOfX));
        rebind(inWidthOfX);
        widthOfX = inWidthOfX;
    }

    /**
     * @brief We need to support assigning the previous state
     */
    template <class OtherHandle>
    LogRegrIRLSTransitionState &operator=(
        const LogRegrIRLSTransitionState<OtherHandle> &inOtherState) {

        for (size_t i = 0; i < mStorage.size(); i++)
            mStorage[i] = inOtherState.mStorage[i];
        return *this;
    }

    /**
     * @brief Merge with another State object by copying the intra-iteration
     *     fields
     */
    template <class OtherHandle>
    LogRegrIRLSTransitionState &operator+=(
        const LogRegrIRLSTransitionState<OtherHandle> &inOtherState) {

        if (mStorage.size() != inOtherState.mStorage.size() ||
            widthOfX != inOtherState.widthOfX)
            throw std::logic_error("Internal error: Incompatible transition "
                                   "states");

        numRows += inOtherState.numRows;
        X_transp_Az += inOtherState.X_transp_Az;
        X_transp_AX += inOtherState.X_transp_AX;
        logLikelihood += inOtherState.logLikelihood;
        // merged state should have the higher status
        // (see top of file for more on 'status' )
        status = (inOtherState.status > status) ? inOtherState.status : status;
        return *this;
    }

    /**
     * @brief Reset the inter-iteration fields.
     */
    inline void reset() {
        numRows         = 0;
        X_transp_Az.fill(0);
        X_transp_AX.fill(0);
        logLikelihood   = 0;
        status          = IN_PROCESS;
    }

  private:
    static inline uint32_t arraySize(const uint16_t inWidthOfX) {
        return 4 + inWidthOfX * inWidthOfX + 2 * inWidthOfX;
    }

    /**
     * @brief Rebind to a new storage array
     *
     * @param inWidthOfX The number of independent variables.
     *
     * Array layout (iteration refers to one aggregate-function call):
     * Inter-iteration components (updated in final function):
     * - 0: widthOfX (number of coefficients)
     * - 1: coef (vector of coefficients)
     *
     * Intra-iteration components (updated in transition step):
     * - 1 + widthOfX: numRows (number of rows already processed in this iteration)
     * - 2 + widthOfX: X_transp_Az (X^T A z)
     * - 2 + 2 * widthOfX: X_transp_AX (X^T A X)
     * - 2 + widthOfX^2 + 2 * widthOfX: logLikelihood ( ln(l(c)) )
     */
    void rebind(uint16_t inWidthOfX = 0) {
        widthOfX.rebind(&mStorage[0]);
        coef.rebind(&mStorage[1], inWidthOfX);
        numRows.rebind(&mStorage[1 + inWidthOfX]);
        X_transp_Az.rebind(&mStorage[2 + inWidthOfX], inWidthOfX);
        X_transp_AX.rebind(&mStorage[2 + 2 * inWidthOfX], inWidthOfX, inWidthOfX);
        logLikelihood.rebind(&mStorage[2 + inWidthOfX * inWidthOfX + 2 * inWidthOfX]);
        status.rebind(&mStorage[3 + inWidthOfX * inWidthOfX + 2 * inWidthOfX]);
    }

    Handle mStorage;

  public:
    typename HandleTraits<Handle>::ReferenceToUInt16 widthOfX;
    typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap coef;

    typename HandleTraits<Handle>::ReferenceToUInt64 numRows;
    typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap X_transp_Az;
    typename HandleTraits<Handle>::MatrixTransparentHandleMap X_transp_AX;
    typename HandleTraits<Handle>::ReferenceToDouble logLikelihood;
    typename HandleTraits<Handle>::ReferenceToUInt16 status;

};


AnyType logregr_irls_step_transition::run(AnyType &args) {
    LogRegrIRLSTransitionState<MutableArrayHandle<double> > state = args[0];
    if (args[1].isNull() || args[2].isNull()) { return args[0]; }
    double y = args[1].getAs<bool>() ? 1. : -1.;
    MappedColumnVector x;
    try {
        // an exception is raised in the backend if args[2] contains nulls
        MappedColumnVector xx = args[2].getAs<MappedColumnVector>();
        // x is a const reference, we can only rebind to change its pointer
        x.rebind(xx.memoryHandle(), xx.size());
    } catch (const ArrayWithNullException &e) {
        return args[0];
    }

    // The following check was added with MADLIB-138.
    if (!x.is_finite()){
        //throw std::domain_error("Design matrix is not finite.");
        warning("Design matrix is not finite.");
        state.status = TERMINATED;
        return state;
    }

    if (state.numRows == 0) {
        if (x.size() > std::numeric_limits<uint16_t>::max()){
            //throw std::domain_error(
            //    "Number of independent variables cannot be larger than 65535.");
            warning("Number of independent variables cannot be larger than 65535.");
            state.status = TERMINATED;
            return state;
        }

        state.initialize(*this, static_cast<uint16_t>(x.size()));
        if (!args[3].isNull()) {
            LogRegrIRLSTransitionState<ArrayHandle<double> > previousState = args[3];

            state = previousState;
            state.reset();
        }
    }

    // Now do the transition step
    state.numRows++;

    // xc = x^T_i c
    double xc = dot(x, state.coef);

    // a_i = sigma(x_i c) sigma(-x_i c)
    double a = sigma(xc) * sigma(-xc);

    // Note: sigma(-x) = 1 - sigma(x).
    //
    //             sigma(-y_i x_i c) y_i
    // z = x_i c + ---------------------
    //                     a_i
    //
    // To avoid overflows if a_i is close to 0, we do not compute z directly,
    // but instead compute a * z.
    double az = xc * a + sigma(-y * xc) * y;

    state.X_transp_Az.noalias() += x * az;
    //triangularView<Lower>(state.X_transp_AX) += x * trans(x) * a;
    state.X_transp_AX += x * trans(x) * a;

    //          n
    //         --
    // l(c) = -\  ln(1 + exp(-y_i * c^T x_i))
    //         /_
    //         i=1
    state.logLikelihood -= std::log( 1. + std::exp(-y * xc) );
    return state;
}


/**
 * @brief Perform the perliminary aggregation function: Merge transition states
 */
AnyType logregr_irls_step_merge_states::run(AnyType &args) {
    LogRegrIRLSTransitionState<MutableArrayHandle<double> > stateLeft = args[0];
    LogRegrIRLSTransitionState<ArrayHandle<double> > stateRight = args[1];

    // We first handle the trivial case where this function is called with one
    // of the states being the initial state
    if (stateLeft.numRows == 0)
        return stateRight;
    else if (stateRight.numRows == 0)
        return stateLeft;

    // Merge states together and return
    stateLeft += stateRight;
    return stateLeft;
}

/**
 * @brief Perform the logistic-regression final step
 */
AnyType logregr_irls_step_final::run(AnyType &args) {
    // We request a mutable object. Depending on the backend, this might perform
    // a deep copy.
    LogRegrIRLSTransitionState<MutableArrayHandle<double> > state = args[0];

    // Aggregates that haven't seen any data just return Null.
    if (state.numRows == 0){
        state.status = NULL_EMPTY;
        return state;
    }

    // See MADLIB-138. At least on certain platforms and with certain versions,
    // LAPACK will run into an infinite loop if pinv() is called for non-finite
    // matrices. We extend the check also to the dependent variables.
    if (!state.X_transp_AX.is_finite() || !state.X_transp_Az.is_finite()){
        //throw NoSolutionFoundException(
        //    "Over- or underflow in intermediate calculation. Input data is "
        //    "likely of poor numerical condition.");
        warning("Over- or underflow in intermediate calculation. Input data is "
              "likely of poor numerical condition.");
        state.status = TERMINATED;
        return state;
    }

    SymmetricPositiveDefiniteEigenDecomposition<Matrix> decomposition(
        state.X_transp_AX, EigenvaluesOnly, ComputePseudoInverse);

    // Precompute (X^T * A * X)^+
    Matrix inverse_of_X_transp_AX = decomposition.pseudoInverse();

    state.coef.noalias() = inverse_of_X_transp_AX * state.X_transp_Az;
    if(!state.coef.is_finite()){
        //throw NoSolutionFoundException(
        //    "Over- or underflow in Newton step, while updating coefficients. "
        //    "Input data is likely of poor numerical condition.");
        warning("Over- or underflow in Newton step, while updating coefficients."
              "Input data is likely of poor numerical condition.");
        state.status = TERMINATED;
        return state;
    }

    // We use the intra-iteration field X_transp_Az for storing the diagonal
    // of X^T A X, so that we don't have to recompute it in the result function.
    // Likewise, we store the condition number.
    // FIXME: This feels a bit like a hack.
    // state.X_transp_Az = inverse_of_X_transp_AX.diagonal();
    // state.X_transp_AX(0,0) = decomposition.conditionNo();
    // state.X_transp_Az(0) = decomposition.conditionNo();
    return state;
}


/**
 * @brief Return the difference in log-likelihood between two states
 */
AnyType internal_logregr_irls_step_distance::run(AnyType &args) {
    LogRegrIRLSTransitionState<ArrayHandle<double> > stateLeft = args[0];
    LogRegrIRLSTransitionState<ArrayHandle<double> > stateRight = args[1];

    if(stateLeft.status == NULL_EMPTY || stateRight.status == NULL_EMPTY){
        return 0.0;
    }

    return std::abs(stateLeft.logLikelihood - stateRight.logLikelihood);
}


/**
 * @brief Return the coefficients and diagnostic statistics of the state
 */
AnyType internal_logregr_irls_result::run(AnyType &args) {
    LogRegrIRLSTransitionState<ArrayHandle<double> > state = args[0];

    if (state.status == NULL_EMPTY)
        return Null();

    return stateToResult(*this, state.coef, state.X_transp_AX,
                         state.logLikelihood,
                         state.status, state.numRows);
}

/**
 * @brief Inter- and intra-iteration state for incremental gradient
 *        method for logistic regression
 *
 * TransitionState encapsualtes the transition state during the
 * logistic-regression aggregate function. To the database, the state is
 * exposed as a single DOUBLE PRECISION array, to the C++ code it is a proper
 * object containing scalars, a vector, and a matrix.
 *
 * Note: We assume that the DOUBLE PRECISION array is initialized by the
 * database with length at least 4, and all elemenets are 0.
 */
template <class Handle>
class LogRegrIGDTransitionState {
    template <class OtherHandle>
    friend class LogRegrIGDTransitionState;

  public:
    LogRegrIGDTransitionState(const AnyType &inArray)
        : mStorage(inArray.getAs<Handle>()) {

        rebind(static_cast<uint16_t>(mStorage[0]));
    }

    /**
     * @brief Convert to backend representation
     *
     * We define this function so that we can use State in the
     * argument list and as a return type.
     */
    inline operator AnyType() const {
        return mStorage;
    }

    /**
     * @brief Initialize the conjugate-gradient state.
     *
     * This function is only called for the first iteration, for the first row.
     */
    inline void initialize(const Allocator &inAllocator, uint16_t inWidthOfX) {
        mStorage = inAllocator.allocateArray<double, dbal::AggregateContext,
                                             dbal::DoZero, dbal::ThrowBadAlloc>(arraySize(inWidthOfX));
        rebind(inWidthOfX);
        widthOfX = inWidthOfX;
    }

    /**
     * @brief We need to support assigning the previous state
     */
    template <class OtherHandle>
    LogRegrIGDTransitionState &operator=(
        const LogRegrIGDTransitionState<OtherHandle> &inOtherState) {

        for (size_t i = 0; i < mStorage.size(); i++)
            mStorage[i] = inOtherState.mStorage[i];
        return *this;
    }

    /**
     * @brief Merge with another State object by copying the intra-iteration
     *     fields
     */
    template <class OtherHandle>
    LogRegrIGDTransitionState &operator+=(
        const LogRegrIGDTransitionState<OtherHandle> &inOtherState) {

        if (mStorage.size() != inOtherState.mStorage.size() ||
            widthOfX != inOtherState.widthOfX)
            throw std::logic_error("Internal error: Incompatible transition "
                                   "states");

        // Compute the average of the models. Note: The following remains an
        // invariant, also after more than one merge:
        // The model is a linear combination of the per-segment models
        // where the coefficient (weight) for each per-segment model is the
        // ratio "# rows in segment / total # rows of all segments merged so
        // far".
        double totalNumRows = static_cast<double>(numRows)
            + static_cast<double>(inOtherState.numRows);
        coef = double(numRows) / totalNumRows * coef
            + double(inOtherState.numRows) / totalNumRows * inOtherState.coef;

        numRows += inOtherState.numRows;
        X_transp_AX += inOtherState.X_transp_AX;
        logLikelihood += inOtherState.logLikelihood;
        // merged state should have the higher status
        // (see top of file for more on 'status' )
        status = (inOtherState.status == TERMINATED) ? inOtherState.status : status;
        return *this;
    }

    /**
     * @brief Reset the inter-iteration fields.
     */
    inline void reset() {
        // FIXME: HAYING: stepsize is hard-coded here now
        stepsize = .01;
        numRows = 0;
        X_transp_AX.fill(0);
        logLikelihood = 0;
        status = IN_PROCESS;
    }

  private:
    static inline uint32_t arraySize(const uint16_t inWidthOfX) {
        return 5 + inWidthOfX * inWidthOfX + inWidthOfX;
    }
    /**
     * @brief Rebind to a new storage array
     *
     * @param inWidthOfX The number of independent variables.
     *
     * Array layout (iteration refers to one aggregate-function call):
     * Inter-iteration components (updated in final function):
     * - 0: widthOfX (number of coefficients)
     * - 1: stepsize (step size of gradient steps)
     * - 2: coef (vector of coefficients)
     *
     * Intra-iteration components (updated in transition step):
     * - 2 + widthOfX: numRows (number of rows already processed in this iteration)
     * - 3 + widthOfX: X_transp_AX (X^T A X)
     * - 3 + widthOfX * widthOfX + widthOfX: logLikelihood ( ln(l(c)) )
     */
    void rebind(uint16_t inWidthOfX) {
        widthOfX.rebind(&mStorage[0]);
        stepsize.rebind(&mStorage[1]);
        coef.rebind(&mStorage[2], inWidthOfX);
        numRows.rebind(&mStorage[2 + inWidthOfX]);
        X_transp_AX.rebind(&mStorage[3 + inWidthOfX], inWidthOfX, inWidthOfX);
        logLikelihood.rebind(&mStorage[3 + inWidthOfX * inWidthOfX + inWidthOfX]);
        status.rebind(&mStorage[4 + inWidthOfX * inWidthOfX + inWidthOfX]);
    }

    Handle mStorage;

  public:
    typename HandleTraits<Handle>::ReferenceToUInt16 widthOfX;
    typename HandleTraits<Handle>::ReferenceToDouble stepsize;
    typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap coef;

    typename HandleTraits<Handle>::ReferenceToUInt64 numRows;
    typename HandleTraits<Handle>::MatrixTransparentHandleMap X_transp_AX;
    typename HandleTraits<Handle>::ReferenceToDouble logLikelihood;
    typename HandleTraits<Handle>::ReferenceToUInt16 status;
};

AnyType
logregr_igd_step_transition::run(AnyType &args) {
    LogRegrIGDTransitionState<MutableArrayHandle<double> > state = args[0];
    if (args[1].isNull() || args[2].isNull()) { return args[0]; }
    double y = args[1].getAs<bool>() ? 1. : -1.;
    MappedColumnVector x;
    try {
        // an exception is raised in the backend if args[2] contains nulls
        MappedColumnVector xx = args[2].getAs<MappedColumnVector>();
        // x is a const reference, we can only rebind to change its pointer
        x.rebind(xx.memoryHandle(), xx.size());
    } catch (const ArrayWithNullException &e) {
        return args[0];
    }

    // The following check was added with MADLIB-138.
    if (!x.is_finite()){
        //throw std::domain_error("Design matrix is not finite.");
        warning("Design matrix is not finite.");
        state.status = TERMINATED;
        return state;
    }

    // We only know the number of independent variables after seeing the first
    // row.
    if (state.numRows == 0) {
        if (x.size() > std::numeric_limits<uint16_t>::max()){
            //throw std::domain_error(
            //    "Number of independent variables cannot be larger than 65535.");
            warning("Number of independent variables cannot be larger than 65535.");
            state.status = TERMINATED;
            return state;
        }

        state.initialize(*this, static_cast<uint16_t>(x.size()));

        // For the first iteration, the previous state is NULL
        if (!args[3].isNull()) {
            LogRegrIGDTransitionState<ArrayHandle<double> > previousState = args[3];
            state = previousState;
            state.reset();
        }
    }

    // Now do the transition step
    state.numRows++;

    // xc = x^T_i c
    double xc = dot(x, state.coef);
    double scale = state.stepsize * sigma(-xc * y) * y;
    state.coef += scale * x;

    // Note: previous coefficients are used for Hessian and log likelihood
    if (!args[3].isNull()) {
        LogRegrIGDTransitionState<ArrayHandle<double> > previousState = args[3];

        double previous_xc = dot(x, previousState.coef);

        // a_i = sigma(x_i c) sigma(-x_i c)
        double a = sigma(previous_xc) * sigma(-previous_xc);
        //triangularView<Lower>(state.X_transp_AX) += x * trans(x) * a;
        state.X_transp_AX += x * trans(x) * a;

        // l_i(c) = - ln(1 + exp(-y_i * c^T x_i))
        state.logLikelihood -= std::log( 1. + std::exp(-y * previous_xc) );
    }

    return state;
}

/**
 * @brief Perform the perliminary aggregation function: Merge transition states
 */
AnyType
logregr_igd_step_merge_states::run(AnyType &args) {
    LogRegrIGDTransitionState<MutableArrayHandle<double> > stateLeft = args[0];
    LogRegrIGDTransitionState<ArrayHandle<double> > stateRight = args[1];

    // We first handle the trivial case where this function is called with one
    // of the states being the initial state
    if (stateLeft.numRows == 0)
        return stateRight;
    else if (stateRight.numRows == 0)
        return stateLeft;

    // Merge states together and return
    stateLeft += stateRight;
    return stateLeft;
}

/**
 * @brief Perform the logistic-regression final step
 *
 * All that we do here is to test whether we have seen any data. If not, we
 * return NULL. Otherwise, we return the transition state unaltered.
 */
AnyType
logregr_igd_step_final::run(AnyType &args) {
    LogRegrIGDTransitionState<MutableArrayHandle<double> > state = args[0];

    if(!state.coef.is_finite()){
        //throw NoSolutionFoundException(
        //    "Overflow or underflow in incremental-gradient iteration. Input "
        //    "data is likely of poor numerical condition.");
        warning("Overflow or underflow in incremental-gradient iteration. Input"
              "data is likely of poor numerical condition.");
        state.status = TERMINATED;
        return state;
    }

    // Aggregates that haven't seen any data just return Null.
    if (state.numRows == 0){
        state.status = NULL_EMPTY;
        return state;
    }

    return state;
}

/**
 * @brief Return the difference in log-likelihood between two states
 */
AnyType
internal_logregr_igd_step_distance::run(AnyType &args) {
    LogRegrIGDTransitionState<ArrayHandle<double> > stateLeft = args[0];
    LogRegrIGDTransitionState<ArrayHandle<double> > stateRight = args[1];

    if(stateLeft.status == NULL_EMPTY || stateRight.status == NULL_EMPTY){
        return 0.0;
    }

    return std::abs(stateLeft.logLikelihood - stateRight.logLikelihood);
}

/**
 * @brief Return the coefficients and diagnostic statistics of the state
 */
AnyType
internal_logregr_igd_result::run(AnyType &args) {
    LogRegrIGDTransitionState<ArrayHandle<double> > state = args[0];

    if (state.status == NULL_EMPTY)
        return Null();

    SymmetricPositiveDefiniteEigenDecomposition<Matrix> decomposition(
        state.X_transp_AX, EigenvaluesOnly, ComputePseudoInverse);

    return stateToResult(*this, state.coef,
                         state.X_transp_AX,
                         state.logLikelihood,
                         state.status, state.numRows);
}

/**
 * @brief Compute the diagnostic statistics
 *
 * This function wraps the common parts of computing the results for both the
 * CG and the IRLS method.
 */
AnyType stateToResult(
    const Allocator &inAllocator,
    const HandleMap<const ColumnVector, TransparentHandle<double> > &inCoef,
    const Matrix & hessian,
    const double &logLikelihood,
    int status,
    const uint64_t &numRows) {

    SymmetricPositiveDefiniteEigenDecomposition<Matrix> decomposition(
        hessian, EigenvaluesOnly, ComputePseudoInverse);

    const Matrix &inverse_of_X_transp_AX = decomposition.pseudoInverse();
    const ColumnVector &diagonal_of_X_transp_AX = inverse_of_X_transp_AX.diagonal();

    MutableNativeColumnVector stdErr(
        inAllocator.allocateArray<double>(inCoef.size()));
    MutableNativeColumnVector waldZStats(
        inAllocator.allocateArray<double>(inCoef.size()));
    MutableNativeColumnVector waldPValues(
        inAllocator.allocateArray<double>(inCoef.size()));
    MutableNativeColumnVector oddsRatios(
        inAllocator.allocateArray<double>(inCoef.size()));

    for (Index i = 0; i < inCoef.size(); ++i) {
        stdErr(i) = std::sqrt(diagonal_of_X_transp_AX(i));
        waldZStats(i) = inCoef(i) / stdErr(i);
        waldPValues(i) = 2. * prob::cdf( prob::normal(),
                                         -std::abs(waldZStats(i)));
        oddsRatios(i) = std::exp( inCoef(i) );
    }

    // Return all coefficients, standard errors, etc. in a tuple
    AnyType tuple;
    tuple << inCoef << logLikelihood << stdErr << waldZStats << waldPValues
          << oddsRatios << inverse_of_X_transp_AX
          << sqrt(decomposition.conditionNo()) << status << numRows;
    return tuple;
}

// ---------------------------------------------------------------------------
//             Robust Logistic Regression States
// ---------------------------------------------------------------------------
/**
 * @brief Inter-and intra-iteration state for robust variance calculation for
 *        logistic regression
 *
 * TransitionState encapsualtes the transition state during the
 * logistic-regression aggregate function. To the database, the state is
 * exposed as a single DOUBLE PRECISION array, to the C++ code it is a proper
 * object containing scalars and vectors.
 *
 * Note: We assume that the DOUBLE PRECISION array is initialized by the
 * database with length at least 5, and all elemenets are 0.
 *
 */
template <class Handle>
class RobustLogRegrTransitionState {
    template <class OtherHandle>
    friend class RobustLogRegrTransitionState;

  public:
    RobustLogRegrTransitionState(const AnyType &inArray)
        : mStorage(inArray.getAs<Handle>()) {

        rebind(static_cast<uint16_t>(mStorage[1]));
    }

    /**
     * @brief Convert to backend representation
     *
     * We define this function so that we can use State in the
     * argument list and as a return type.
     */
    inline operator AnyType() const {
        return mStorage;
    }

    /**
     * @brief Initialize the robust variance calculation state.
     *
     * This function is only called for the first iteration, for the first row.
     */
    inline void initialize(const Allocator &inAllocator, uint16_t inWidthOfX) {
        mStorage = inAllocator.allocateArray<double, dbal::AggregateContext,
                                             dbal::DoZero, dbal::ThrowBadAlloc>(arraySize(inWidthOfX));
        rebind(inWidthOfX);
        widthOfX = inWidthOfX;
    }

    /**
     * @brief We need to support assigning the previous state
     */
    template <class OtherHandle>
    RobustLogRegrTransitionState &operator=(
        const RobustLogRegrTransitionState<OtherHandle> &inOtherState) {

        for (size_t i = 0; i < mStorage.size(); i++)
            mStorage[i] = inOtherState.mStorage[i];
        return *this;
    }

    /**
     * @brief Merge with another State object by copying the intra-iteration
     *     fields
     */
    template <class OtherHandle>
    RobustLogRegrTransitionState &operator+=(
        const RobustLogRegrTransitionState<OtherHandle> &inOtherState) {

        if (mStorage.size() != inOtherState.mStorage.size() ||
            widthOfX != inOtherState.widthOfX)
            throw std::logic_error("Internal error: Incompatible transition "
                                   "states");

        numRows += inOtherState.numRows;
        X_transp_AX += inOtherState.X_transp_AX;
        meat += inOtherState.meat;
        return *this;
    }

    /**
     * @brief Reset the inter-iteration fields.
     */
    inline void reset() {
        numRows = 0;
        X_transp_AX.fill(0);
        meat.fill(0);

    }

  private:
    static inline size_t arraySize(const uint16_t inWidthOfX) {
        return 4 + 2 * inWidthOfX * inWidthOfX + inWidthOfX;
    }

    /**
     * @brief Rebind to a new storage array
     *
     * @param inWidthOfX The number of independent variables.
     *
     * Array layout (variables that are constant throughout function call):
     * Inter-iteration components
     * - 0: Iteration (What iteration is this)
     * - 1: widthOfX (number of coefficients)
     * - 2: coef (vector of coefficients)
     *
     * Intra-iteration components (variables that updated in transition step):
     * - 2 + widthOfX: numRows (number of rows already processed in this iteration)
     * - 3 + widthOfX: X_transp_AX (X^T A X)
     * - 3 + widthOfX * widthOfX + widthOfX: meat (the meat matrix)
     * - 3 + 2 * widthOfX * widthOfX + widthOfX: grad (intermediate value for gradient)
     */
    void rebind(uint16_t inWidthOfX) {
        iteration.rebind(&mStorage[0]);
        widthOfX.rebind(&mStorage[1]);
        coef.rebind(&mStorage[2], inWidthOfX);
        numRows.rebind(&mStorage[2 + inWidthOfX]);
        X_transp_AX.rebind(&mStorage[3 + inWidthOfX], inWidthOfX, inWidthOfX);
        meat.rebind(&mStorage[3 + inWidthOfX * inWidthOfX + inWidthOfX], inWidthOfX, inWidthOfX);
    }

    Handle mStorage;

  public:
    typename HandleTraits<Handle>::ReferenceToUInt32 iteration;
    typename HandleTraits<Handle>::ReferenceToUInt16 widthOfX;
    typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap coef;

    typename HandleTraits<Handle>::ReferenceToUInt64 numRows;
    typename HandleTraits<Handle>::MatrixTransparentHandleMap X_transp_AX;
    typename HandleTraits<Handle>::MatrixTransparentHandleMap meat;
};



/**
 * @brief Helper function that computes the final statistics for the robust variance
 */

AnyType robuststateToResult(
    const Allocator &inAllocator,
    const ColumnVector &inCoef,
    const ColumnVector &diagonal_of_varianceMat) {

    MutableNativeColumnVector variance(
        inAllocator.allocateArray<double>(inCoef.size()));

    MutableNativeColumnVector coef(
        inAllocator.allocateArray<double>(inCoef.size()));

    MutableNativeColumnVector stdErr(
        inAllocator.allocateArray<double>(inCoef.size()));
    MutableNativeColumnVector waldZStats(
        inAllocator.allocateArray<double>(inCoef.size()));
    MutableNativeColumnVector waldPValues(
        inAllocator.allocateArray<double>(inCoef.size()));

    for (Index i = 0; i < inCoef.size(); ++i) {
        //variance(i) = diagonal_of_varianceMat(i);
        coef(i) = inCoef(i);

        stdErr(i) = std::sqrt(diagonal_of_varianceMat(i));
        waldZStats(i) = inCoef(i) / stdErr(i);
        waldPValues(i) = 2. * prob::cdf(
            prob::normal(), -std::abs(waldZStats(i)));
    }

    // Return all coefficients, standard errors, etc. in a tuple
    AnyType tuple;
    //tuple <<  variance<<stdErr << waldZStats << waldPValues;
    tuple <<  coef<<stdErr << waldZStats << waldPValues;
    return tuple;
}

/**
 * @brief Perform the logistic-regression transition step
 */
AnyType
robust_logregr_step_transition::run(AnyType &args) {
    // Early return because of an exception has been "thrown"
    // (actually "warning") in the previous invocations
    if(args[0].isNull())
        return Null();
    RobustLogRegrTransitionState<MutableArrayHandle<double> > state = args[0];
    if (args[1].isNull() || args[2].isNull()) { return args[0]; }
    double y = args[1].getAs<bool>() ? 1. : -1.;
    MappedColumnVector x;
    try {
        // an exception is raised in the backend if args[2] contains nulls
        MappedColumnVector xx = args[2].getAs<MappedColumnVector>();
        // x is a const reference, we can only rebind to change its pointer
        x.rebind(xx.memoryHandle(), xx.size());
    } catch (const ArrayWithNullException &e) {
        return args[0];
    }
    MappedColumnVector coef = args[3].getAs<MappedColumnVector>();

    // The following check was added with MADLIB-138.
    if (!dbal::eigen_integration::isfinite(x)) {
        //throw std::domain_error("Design matrix is not finite.");
        warning("Design matrix is not finite.");
        return Null();
    }

    if (state.numRows == 0) {
        if (x.size() > std::numeric_limits<uint16_t>::max()) {
            //throw std::domain_error("Number of independent variables cannot be "
            //                        "larger than 65535.");
            warning("Number of independent variables cannot be larger than 65535.");
            return Null();
        }

        state.initialize(*this, static_cast<uint16_t>(x.size()));
        state.coef = coef; //Copy this into the state for later
    }

    // Now do the transition step
    state.numRows++;
    double xc = dot(x, coef);
    ColumnVector Grad;
    Grad = sigma(-y * xc) * y * trans(x);

    Matrix GradGradTranspose;
    GradGradTranspose = Grad*Grad.transpose();
    state.meat += GradGradTranspose;

    // Note: sigma(-x) = 1 - sigma(x).
    // a_i = sigma(x_i c) sigma(-x_i c)
    double a = sigma(xc) * sigma(-xc);
    triangularView<Lower>(state.X_transp_AX) += x * trans(x) * a;
    return state;
}


/**
 * @brief Perform the perliminary aggregation function: Merge transition states
 */
AnyType
robust_logregr_step_merge_states::run(AnyType &args) {
    // In case the aggregator should be terminated because
    // an exception has been "thrown" in the transition function
    if(args[0].isNull() || args[1].isNull())
        return Null();

    RobustLogRegrTransitionState<MutableArrayHandle<double> > stateLeft = args[0];
    RobustLogRegrTransitionState<ArrayHandle<double> > stateRight = args[1];
    // We first handle the trivial case where this function is called with one
    // of the states being the initial state
    if (stateLeft.numRows == 0)
        return stateRight;
    else if (stateRight.numRows == 0)
        return stateLeft;

    // Merge states together and return
    stateLeft += stateRight;
    return stateLeft;
}

/**
 * @brief Perform the robust variance calculation for logistic-regression final step
 */
AnyType
robust_logregr_step_final::run(AnyType &args) {
    // In case the aggregator should be terminated because
    // an exception has been "thrown" in the transition function
    if (args[0].isNull())
        return Null();
    // We request a mutable object. Depending on the backend, this might perform
    // a deep copy.
    RobustLogRegrTransitionState<MutableArrayHandle<double> > state = args[0];
    // Aggregates that haven't seen any data just return Null.
    if (state.numRows == 0)
        return Null();

    //Compute the robust variance with the White sandwich estimator
    SymmetricPositiveDefiniteEigenDecomposition<Matrix> decomposition(
        state.X_transp_AX, EigenvaluesOnly, ComputePseudoInverse);

    Matrix bread = decomposition.pseudoInverse();

    /*
      This is written a little strangely because it prevents Eigen warnings.
      The following two lines are equivalent to:
      Matrix variance = bread*state.meat*bread;
      but eigen throws a warning on that.
    */
    Matrix varianceMat;// = meat;
    varianceMat = bread*state.meat*bread;

    /*
     * Computing the results for robust variance
     */

    return robuststateToResult(*this, state.coef,
                               varianceMat.diagonal());
}

// ------------------------ End of Robust ------------------------------------




// ---------------------------------------------------------------------------
//             Marginal Effects Logistic Regression States
// ---------------------------------------------------------------------------
/**
 * @brief State for marginal effects calculation for logistic regression
 *
 * TransitionState encapsualtes the transition state during the
 * marginal effects calculation for the logistic-regression aggregate function.
 * To the database, the state is exposed as a single DOUBLE PRECISION array,
 * to the C++ code it is a proper object containing scalars and vectors.
 *
 * Note: We assume that the DOUBLE PRECISION array is initialized by the
 * database with length at least 5, and all elemenets are 0.
 *
 */
template <class Handle>
class MarginalLogRegrTransitionState {
    template <class OtherHandle>
    friend class MarginalLogRegrTransitionState;

  public:
    MarginalLogRegrTransitionState(const AnyType &inArray)
        : mStorage(inArray.getAs<Handle>()) {

        rebind(static_cast<uint16_t>(mStorage[1]));
    }

    /**
     * @brief Convert to backend representation
     *
     * We define this function so that we can use State in the
     * argument list and as a return type.
     */
    inline operator AnyType() const {
        return mStorage;
    }

    /**
     * @brief Initialize the marginal variance calculation state.
     *
     * This function is only called for the first iteration, for the first row.
     */
    inline void initialize(const Allocator &inAllocator, uint16_t inWidthOfX) {
        mStorage = inAllocator.allocateArray<double, dbal::AggregateContext,
                                             dbal::DoZero, dbal::ThrowBadAlloc>(arraySize(inWidthOfX));
        rebind(inWidthOfX);
        widthOfX = inWidthOfX;
    }

    /**
     * @brief We need to support assigning the previous state
     */
    template <class OtherHandle>
    MarginalLogRegrTransitionState &operator=(
        const MarginalLogRegrTransitionState<OtherHandle> &inOtherState) {

        for (size_t i = 0; i < mStorage.size(); i++)
            mStorage[i] = inOtherState.mStorage[i];
        return *this;
    }

    /**
     * @brief Merge with another State object by copying the intra-iteration
     *     fields
     */
    template <class OtherHandle>
    MarginalLogRegrTransitionState &operator+=(
        const MarginalLogRegrTransitionState<OtherHandle> &inOtherState) {

        if (mStorage.size() != inOtherState.mStorage.size() ||
            widthOfX != inOtherState.widthOfX)
            throw std::logic_error("Internal error: Incompatible transition "
                                   "states");

        numRows += inOtherState.numRows;
        marginal_effects_per_observation += inOtherState.marginal_effects_per_observation;
        X_bar += inOtherState.X_bar;
        X_transp_AX += inOtherState.X_transp_AX;
        delta += inOtherState.delta;
        return *this;
    }

    /**
     * @brief Reset the inter-iteration fields.
     */
    inline void reset() {
        numRows = 0;
        marginal_effects_per_observation = 0;
        X_bar.fill(0);
        X_transp_AX.fill(0);
        delta.fill(0);
    }

  private:
    static inline size_t arraySize(const uint16_t inWidthOfX) {
        return 4 + 2 * inWidthOfX * inWidthOfX + 2 * inWidthOfX;
    }

    /**
     * @brief Rebind to a new storage array
     *
     * @param inWidthOfX The number of independent variables.
     *
     * Array layout (variables that are constant throughout function call):
     * Inter-iteration components
     * - 0: Iteration (What iteration is this)
     * - 1: widthOfX (number of coefficients)
     * - 2: coef (vector of coefficients)
     *
     * Intra-iteration components (variables that updated in transition step):
     * - 2 + widthOfX: numRows (number of rows already processed in this iteration)
     * - 3 + widthOfX: X_transp_AX (X^T A X)
     */
    void rebind(uint16_t inWidthOfX) {
        iteration.rebind(&mStorage[0]);
        widthOfX.rebind(&mStorage[1]);
        coef.rebind(&mStorage[2], inWidthOfX);
        numRows.rebind(&mStorage[2 + inWidthOfX]);
        marginal_effects_per_observation.rebind(&mStorage[3 + inWidthOfX]);
        X_bar.rebind(&mStorage[4 + inWidthOfX], inWidthOfX);
        X_transp_AX.rebind(&mStorage[4 + 2*inWidthOfX], inWidthOfX, inWidthOfX);
        delta.rebind(&mStorage[4+inWidthOfX*inWidthOfX+2*inWidthOfX], inWidthOfX, inWidthOfX);
    }
    Handle mStorage;

  public:

    typename HandleTraits<Handle>::ReferenceToUInt32 iteration;
    typename HandleTraits<Handle>::ReferenceToUInt16 widthOfX;
    typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap coef;
    typename HandleTraits<Handle>::ReferenceToUInt64 numRows;
    typename HandleTraits<Handle>::ReferenceToDouble marginal_effects_per_observation;

    typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap X_bar;
    typename HandleTraits<Handle>::MatrixTransparentHandleMap X_transp_AX;
    typename HandleTraits<Handle>::MatrixTransparentHandleMap delta;
};

// ----------------------------------------------------------------------

/**
 * @brief Helper function that computes the final statistics for the marginal variance
 */

AnyType marginalstateToResult(
    const Allocator &inAllocator,
    const ColumnVector &inCoef,
    const ColumnVector &diagonal_of_variance_matrix,
    const double inmarginal_effects_per_observation,
    const Index numRows) {

    MutableNativeColumnVector marginal_effects(
        inAllocator.allocateArray<double>(inCoef.size()));
    MutableNativeColumnVector coef(
        inAllocator.allocateArray<double>(inCoef.size()));
    MutableNativeColumnVector stdErr(
        inAllocator.allocateArray<double>(inCoef.size()));
    MutableNativeColumnVector tStats(
        inAllocator.allocateArray<double>(inCoef.size()));
    MutableNativeColumnVector pValues(
        inAllocator.allocateArray<double>(inCoef.size()));

    for (Index i = 0; i < inCoef.size(); ++i) {
        coef(i) = inCoef(i);
        marginal_effects(i) = inCoef(i) * inmarginal_effects_per_observation / static_cast<double>(numRows);
        stdErr(i) = std::sqrt(diagonal_of_variance_matrix(i));
        tStats(i) = marginal_effects(i) / stdErr(i);

        // P-values only make sense if numRows > coef.size()
        if (numRows > inCoef.size())
            pValues(i) = 2. * prob::cdf(
                prob::normal(), -std::abs(tStats(i)));
    }

    // Return all coefficients, standard errors, etc. in a tuple
    // Note: PValues will return NULL if numRows <= coef.size
    AnyType tuple;
    tuple << marginal_effects
          << coef
          << stdErr
          << tStats
          << (numRows > inCoef.size()? pValues: Null());
    return tuple;
}


/**
 * @brief Perform the marginal effects transition step
 */
AnyType
marginal_logregr_step_transition::run(AnyType &args) {
    // Early return because of an exception has been "thrown"
    // (actually "warning") in the previous invocations
    if (args[0].isNull())
        return Null();
    MarginalLogRegrTransitionState<MutableArrayHandle<double> > state = args[0];
    if (args[1].isNull() || args[2].isNull()) { return args[0]; }
    MappedColumnVector x;
    try {
        // an exception is raised in the backend if args[2] contains nulls
        MappedColumnVector xx = args[2].getAs<MappedColumnVector>();
        // x is a const reference, we can only rebind to change its pointer
        x.rebind(xx.memoryHandle(), xx.size());
    } catch (const ArrayWithNullException &e) {
        return args[0];
    }

    MappedColumnVector coef = args[3].getAs<MappedColumnVector>();

    // The following check was added with MADLIB-138.
    if (!dbal::eigen_integration::isfinite(x)) {
        //throw std::domain_error("Design matrix is not finite.");
        warning("Design matrix is not finite.");
        return Null();
    }

    if (state.numRows == 0) {
        if (x.size() > std::numeric_limits<uint16_t>::max()) {
            //throw std::domain_error("Number of independent variables cannot be "
            //                        "larger than 65535.");
            warning("Number of independent variables cannot be larger than 65535.");
            return Null();
        }
        state.initialize(*this, static_cast<uint16_t>(x.size()));
        state.coef = coef; //Copy this into the state for later
    }

    // Now do the transition step
    state.numRows++;
    double xc = dot(x, coef);
    double p = std::exp(xc)/ (1 + std::exp(xc));
    double a = sigma(xc) * sigma(-xc);

    // TODO: Change the average code so it won't overflow
    state.marginal_effects_per_observation += p * (1 - p);
    state.X_bar += x;
    state.X_transp_AX += x * trans(x) * a;

    Matrix delta;
    delta = (1 - 2*p) * state.coef * trans(x);
    // This should be faster than adding an identity
    for (int i=0; i < state.widthOfX; i++){
        delta(i,i) += 1;
    }

    // Standard error according to the delta method
    state.delta += p * (1 - p) * delta;

    return state;
}


/**
 * @brief Marginal effects: Merge transition states
 */
AnyType
marginal_logregr_step_merge_states::run(AnyType &args) {
    // In case the aggregator should be terminated because
    // an exception has been "thrown" in the transition function
    if(args[0].isNull() || args[1].isNull())
        return Null();

    MarginalLogRegrTransitionState<MutableArrayHandle<double> > stateLeft = args[0];
    MarginalLogRegrTransitionState<ArrayHandle<double> > stateRight = args[1];
    // We first handle the trivial case where this function is called with one
    // of the states being the initial state
    if (stateLeft.numRows == 0)
        return stateRight;
    else if (stateRight.numRows == 0)
        return stateLeft;

    // Merge states together and return
    stateLeft += stateRight;
    return stateLeft;
}

/**
 * @brief Marginal effects: Final step
 */
AnyType
marginal_logregr_step_final::run(AnyType &args) {
    // In case the aggregator should be terminated because
    // an exception has been "thrown" in the transition function
    if (args[0].isNull())
        return Null();

    // We request a mutable object.
    // Depending on the backend, this might perform a deep copy.
    MarginalLogRegrTransitionState<MutableArrayHandle<double> > state = args[0];
    // Aggregates that haven't seen any data just return Null.
    if (state.numRows == 0)
        return Null();

    // Compute variance matrix of logistic regression
    SymmetricPositiveDefiniteEigenDecomposition<Matrix> decomposition(
        state.X_transp_AX, EigenvaluesOnly, ComputePseudoInverse);
    Matrix variance = decomposition.pseudoInverse();
    // Standard error according to the delta method
    Matrix std_err;
    std_err = state.delta * variance * trans(state.delta) / static_cast<double>(state.numRows*state.numRows);

    // Computing the marginal effects
    return marginalstateToResult(*this,
                                 state.coef,
                                 std_err.diagonal(),
                                 state.marginal_effects_per_observation,
                                 state.numRows);
}

// ------------------------ End of Marginal ------------------------------------

AnyType logregr_predict::run(AnyType &args) {
    try {
        args[0].getAs<MappedColumnVector>();
    } catch (const ArrayWithNullException &e) {
        throw std::runtime_error(
            "Logregr error: the coefficients contain NULL values");
    }

    // returns NULL if args[1] (features) contains NULL values
    try {
        args[1].getAs<MappedColumnVector>();
    } catch (const ArrayWithNullException &e) {
        return Null();
    }

    MappedColumnVector vec1 = args[0].getAs<MappedColumnVector>();
    MappedColumnVector vec2 = args[1].getAs<MappedColumnVector>();

    if (vec1.size() != vec2.size())
        throw std::runtime_error(
            "Coefficients and independent variables are of incompatible length");

    return vec1.dot(vec2) > 0 ? true : false;
}

AnyType logregr_predict_prob::run(AnyType &args) {
    try {
        args[0].getAs<MappedColumnVector>();
    } catch (const ArrayWithNullException &e) {
        throw std::runtime_error(
            "Logregr error: the coefficients contain NULL values");
    }

    // returns NULL if args[1] (features) contains NULL values
    try {
        args[1].getAs<MappedColumnVector>();
    } catch (const ArrayWithNullException &e) {
        return Null();
    }

    MappedColumnVector vec1 = args[0].getAs<MappedColumnVector>();
    MappedColumnVector vec2 = args[1].getAs<MappedColumnVector>();

    if (vec1.size() != vec2.size())
        throw std::runtime_error(
            "Coefficients and independent variables are of incompatible length");

    double dot = vec1.dot(vec2);
    double logit = 0.0;
    // Underflow/overfolow handling
    try {
        logit = 1.0 / (1 + exp(-dot));
    } catch (...) {
        logit = (dot > 0) ? 1.0 : 0.0;
    }
    return logit;
}

} // namespace regress
} // namespace modules
} // namespace madlib
