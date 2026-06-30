module IndicatorsTemporalExt

using Temporal
using Indicators

# Methods for porting Indicators.jl functions to TS objects from Temporal.jl package
function close_fun(X::TS, f::Function, flds::Vector{Symbol}; args...)
    if size(X,2) == 1
        return TS(f(X.values; args...), X.index, flds)
    elseif size(X,2) > 1 && has_close(X)
        return TS(f(cl(X).values; args...), X.index, flds)
    else
        error("Must be univariate or contain Close/Settle/Last.")
    end
end
function hlc_fun(X::TS, f::Function, flds::Vector{Symbol}; args...)
    if size(X,2) == 3
        return TS(f(X.values; args...), X.index, flds)
    elseif size(X,2) > 3 && has_high(X) && has_low(X) && has_close(X)
        return TS(f(hlc(X).values; args...), X.index, flds)
    else
        error("Argument must have 3 columns or have High, Low, and Close/Settle/Last fields.")
    end
end
function hl_fun(X::TS, f::Function, flds::Vector{Symbol}; args...)
    if size(X,2) == 2
        return TS(f(X.values; args...), X.index, flds)
    elseif size(X,2) > 2 && has_high(X) && has_low(X)
        return TS(f(hl(X).values; args...), X.index, flds)
    else
        error("Argument must have 2 columns or have High and Low fields.")
    end
end
function ohlc_fun(X::TS, f::Function, flds::Vector{Symbol}; args...)
    if size(X,2) == 4
        return TS(f(X.values; args...), X.index, flds)
    elseif size(X,2) > 4 && has_open(X) && has_high(X) && has_low(X) && has_close(X)
        return TS(f(ohlc(X).values; args...), X.index, flds)
    else
        error("Argument must have 4 columns or have Open, High, Low, and Close/Settle/Last fields.")
    end
end
function cv_fun(X::TS, f::Function, flds::Vector{Symbol}; args...)
    if size(X,2) == 2
        return TS(f(X.values; args...), X.index, flds)
    elseif size(X,2) > 2 && has_close(X) && has_volume(X)
        return TS(f([cl(X) vo(X)].values; args...), X.index, flds)
    else
        error("Argument must have 2 columns or have Close and Volume fields.")
    end
end

###### run.jl ######
function Indicators.runcov(x::TS{V,T}, y::TS{V,T}; args...) where {V,T}
    @assert size(x,2) == 1 && size(y,2) == 1 "Arguments x and y must both be univariate (have only one column)."
    z = [x y].values
    return ts(Indicators.runcov(z[:,1], z[:,2]; args...), x.index, :RunCov)
end
function Indicators.runcor(x::TS{V,T}, y::TS{V,T}; args...) where {V,T}
    @assert size(x,2) == 1 && size(y,2) == 1 "Arguments x and y must both be univariate (have only one column)."
    z = [x y].values
    ts(Indicators.runcor(z[:,1], z[:,2]; args...), x.index, :RunCor)
end
Indicators.mode(X::TS) = Indicators.mode(X.values)
Indicators.runmean(X::TS; args...) = close_fun(X, Indicators.runmean, [:RunMean]; args...)
Indicators.runsum(X::TS; args...) = close_fun(X, Indicators.runsum, [:RunSum]; args...)
Indicators.runmad(X::TS; args...) = close_fun(X, Indicators.runmad, [:RunMAD]; args...)
Indicators.runvar(X::TS; args...) = close_fun(X, Indicators.runvar, [:RunVar]; args...)
Indicators.runmax(X::TS; args...) = close_fun(X, Indicators.runmax, [:RunMax]; args...)
Indicators.runmin(X::TS; args...) = close_fun(X, Indicators.runmin, [:RunMin]; args...)
Indicators.runsd(X::TS; args...) = close_fun(X, Indicators.runsd, [:RunSD]; args...)
Indicators.runquantile(X::TS; args...) = close_fun(X, Indicators.runquantile, [:RunQuantile]; args...)
Indicators.wilder_sum(X::TS; args...) = close_fun(X, Indicators.wilder_sum, [:WilderSum]; args...)

# runacf implementation using Temporal.acf
"""
```
function runacf(x::Vector{T};
                n::Int = 10,
                maxlag::Int = n-3,
                lags::AbstractVector{Int,1} = 0:maxlag,
                cumulative::Bool = true)::Matrix{T} where {T<:Real}
                runacf(X::Matrix; n::Int=10, cumulative::Bool=true, maxlag::Int=n-3, lags::AbstractVector{Int}=0:maxlag)::Matrix{Float64}
```

Compute the running/rolling autocorrelation of a vector.
"""
function Indicators.runacf(x::AbstractVector{T};
                n::Int = 10,
                maxlag::Int = n-3,
                lags::AbstractVector{Int} = 0:maxlag,
                cumulative::Bool = true)::Matrix{T} where {T<:Real}
    @assert size(x, 2) == 1 "Autocorrelation input array must be one-dimensional"
    N = size(x, 1)
    @assert n < N && n > 0
    if length(lags) == 1 && lags[1] == 0
        return ones((N, 1))
    end
    out = zeros((N, length(lags))) * NaN
    if cumulative
        @inbounds for i in n:N
            out[i,:] = Temporal.acf(x[1:i], lags=lags)
        end
    else
        @inbounds for i in n:N
            out[i,:] = Temporal.acf(x[i-n+1:i], lags=lags)
        end
    end
    return out
end
Indicators.runacf(X::AbstractMatrix; n::Int=10, cumulative::Bool=true, maxlag::Int=n-3, lags::AbstractVector{Int}=0:maxlag)::Matrix{Float64} = hcat((Indicators.runacf(X[:,j], n=n, cumulative=cumulative, maxlag=maxlag, lags=lags) for j in 1:size(X,2))...)

function Indicators.runacf(X::TS; n::Int=10, maxlag::Int=n-3, lags::AbstractArray{Int,1}=0:maxlag, cumulative::Bool=true)
    close_fun(X, (x; args...) -> Indicators.runacf(x; args...), [Symbol(i) for i in lags]; n=n, maxlag=maxlag, lags=lags, cumulative=cumulative)
end
Indicators.runfun(X::TS, f::Function; n::Int=10, cumulative::Bool=true, args...) = TS(Indicators.runfun(X, f, n=n, cumulative=cumulative, args...), X.index, [:Function])

##### ma.jl ######
Indicators.sma(X::TS; args...) = close_fun(X, Indicators.sma, [:SMA]; args...)
Indicators.hma(X::TS; args...) = close_fun(X, Indicators.hma, [:HMA]; args...)
Indicators.mma(X::TS; args...) = close_fun(X, Indicators.mma, [:MMA]; args...)
Indicators.swma(X::TS; args...) = close_fun(X, Indicators.swma, [:SWMA]; args...)
Indicators.kama(X::TS; args...) = close_fun(X, Indicators.kama, [:KAMA]; args...)
Indicators.alma(X::TS; args...) = close_fun(X, Indicators.alma, [:ALMA]; args...)
Indicators.trima(X::TS; args...) = close_fun(X, Indicators.trima, [:TRIMA]; args...)
Indicators.wma(X::TS; args...) = close_fun(X, Indicators.wma, [:WMA]; args...)
Indicators.ema(X::TS; args...) = close_fun(X, Indicators.ema, [:EMA]; args...)
Indicators.dema(X::TS; args...) = close_fun(X, Indicators.dema, [:DEMA]; args...)
Indicators.tema(X::TS; args...) = close_fun(X, Indicators.tema, [:TEMA]; args...)
Indicators.zlema(X::TS; args...) = close_fun(X, Indicators.zlema, [:ZLEMA]; args...)
Indicators.mama(X::TS; args...) = close_fun(X, Indicators.mama, [:MAMA,:FAMA]; args...)
Indicators.vwma(X::TS; args...) = cv_fun(X, Indicators.vwma, [:VWMA]; args...)
Indicators.vwap(X::TS; args...) = cv_fun(X, Indicators.vwma, [:VWAP]; args...)
Indicators.hama(X::TS; args...) = close_fun(X, Indicators.hama, [:HammingMA]; args...)

##### reg.jl ######
Indicators.mlr_beta(X::TS; args...) = close_fun(X, Indicators.mlr_beta, [:Intercept,:Slope]; args...)
Indicators.mlr_slope(X::TS; args...) = close_fun(X, Indicators.mlr_slope, [:Slope]; args...)
Indicators.mlr_intercept(X::TS; args...) = close_fun(X, Indicators.mlr_intercept, [:Intercept]; args...)
Indicators.mlr(X::TS; args...) = close_fun(X, Indicators.mlr, [:MLR]; args...)
Indicators.mlr_se(X::TS; args...) = close_fun(X, Indicators.mlr_se, [:StdErr]; args...)
Indicators.mlr_ub(X::TS; args...) = close_fun(X, Indicators.mlr_ub, [:MLRUB]; args...)
Indicators.mlr_lb(X::TS; args...) = close_fun(X, Indicators.mlr_lb, [:MLRLB]; args...)
Indicators.mlr_bands(X::TS; args...) = close_fun(X, Indicators.mlr_bands, [:MLRLB,:MLR,:MLRUB]; args...)
Indicators.mlr_rsq(X::TS; args...) = close_fun(X, Indicators.mlr_rsq, [:RSquared]; args...)

##### mom.jl ######
Indicators.momentum(X::TS; args...) = close_fun(X, Indicators.momentum, [:Momentum]; args...)
Indicators.roc(X::TS; args...) = close_fun(X, Indicators.roc, [:ROC]; args...)
Indicators.macd(X::TS; args...) = close_fun(X, Indicators.macd, [:MACD,:Signal,:Histogram]; args...)
Indicators.rsi(X::TS; args...) = close_fun(X, Indicators.rsi, [:RSI]; args...)
Indicators.psar(X::TS; args...) = hl_fun(X, Indicators.psar, [:PSAR]; args...)
Indicators.kst(X::TS; args...) = close_fun(X, Indicators.kst, [:KST]; args...)
Indicators.wpr(X::TS; args...) = hlc_fun(X, Indicators.wpr, [:WPR]; args...)
Indicators.adx(X::TS; args...) = hlc_fun(X, Indicators.adx, [:DiPlus,:DiMinus,:ADX]; args...)
Indicators.heikinashi(X::TS; args...) = ohlc_fun(X, Indicators.heikinashi, [:Open,:High,:Low,:Close]; args...)
Indicators.cci(X::TS; args...) = hlc_fun(X, Indicators.cci, [:CCI]; args...)
Indicators.stoch(X::TS; args...) = hlc_fun(X, Indicators.stoch, [:Stochastic,:Signal]; args...)
Indicators.smi(X::TS; args...) = hlc_fun(X, Indicators.smi, [:SMI,:Signal]; args...)
Indicators.donch(X::TS; args...) = hl_fun(X, Indicators.donch, [:Low,:Mid,:High]; args...)
Indicators.ichimoku(X::TS; args...) = hlc_fun(X, Indicators.ichimoku, [:Tenkan,:Kijun,:SenkouA,:SenkouB,:Chikou]; args...)
Indicators.aroon(X::TS; args...) = hl_fun(X, Indicators.aroon, [:AroonUp,:AroonDn,:AroonOsc]; args...)

##### vol.jl ######
Indicators.bbands(X::TS; args...) = close_fun(X, Indicators.bbands, [:LB,:MA,:UB]; args...)
Indicators.tr(X::TS; args...) = hlc_fun(X, Indicators.tr, [:TR]; args...)
Indicators.atr(X::TS; args...) = hlc_fun(X, Indicators.atr, [:ATR]; args...)
Indicators.keltner(X::TS; args...) = hlc_fun(X, Indicators.keltner, [:KeltnerLower,:KeltnerMiddle,:KeltnerUpper]; args...)

##### trendy.jl #####
Indicators.maxima(X::TS; args...) = close_fun(X, Indicators.maxima, [:Maxima]; args...)
Indicators.minima(X::TS; args...) = close_fun(X, Indicators.minima, [:Minima]; args...)
Indicators.support(X::TS; args...) = close_fun(X, Indicators.support, [:Support]; args...)
Indicators.resistance(X::TS; args...) = close_fun(X, Indicators.resistance, [:Resistance]; args...)

#### utils.jl ####
Indicators.crossover(x::TS, y::TS) = ts(Indicators.crossover(x.values, y.values), x.index, [:CrossOver])
Indicators.crossunder(x::TS, y::TS) = ts(Indicators.crossunder(x.values, y.values), x.index, [:CrossUnder])

#### chaos.jl ####
Indicators.hurst(x::TS; args...) = close_fun(x, Indicators.hurst, [:Hurst]; args...)
Indicators.rsrange(x::TS; args...) = close_fun(x, Indicators.rsrange, [:RS]; args...)

end
