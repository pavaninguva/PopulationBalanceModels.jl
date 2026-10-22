_measureaxis(term,grid)=term.coordinate_map===nothing ? grid.centers : term.coordinate_map.measure.(grid.centers)
_measure(term,v)=term.coordinate_map===nothing ? v : term.coordinate_map.measure(v)

# Return the moments per birth that leave the resolved pivot range.
function _deposit!(dN, centers, v, rate, overflow; on_escape=nothing)
    isfinite(v) && v>=0 && isfinite(rate) && rate>=0 ||
        throw(NumericalFailure("deposition requires a finite nonnegative coordinate and birth rate"))
    if v < first(centers) || v > last(centers)
        if overflow === :error && !iszero(rate)
            throw(NumericalFailure("birth size $v outside pivot range; enlarge grid or explicitly use overflow=:discard"))
        end
        on_escape===nothing || on_escape(v,rate)
        return (rate,rate*v)
    end
    j = searchsortedlast(centers,v)
    if j == length(centers)
        dN[j] += rate
    else
        w = (centers[j+1]-v)/(centers[j+1]-centers[j])
        dN[j] += w*rate
        dN[j+1] += (1-w)*rate
    end
    (zero(rate),zero(rate*v))
end
