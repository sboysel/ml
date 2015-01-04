using RDatasets
using DecisionTree
using MLBase

# [:Id,:Sch,:Gr,:ClType,:HDeg,:CLad,:Exp,:Trace,:Read,:Math,:SES,:SchType,:Sx,:Eth,:BirthQ,:BirthY,:Yrs,:Tch]

function encode_frame(d)
    df = copy(d)
    for col in df.colindex.names
        if eltype(df[col]) == ASCIIString
            lm = labelmap(df[col])
            println("($col) mapped.")
            df[col] = labelencode(lm, df[col])
            println("($col) encoded.")
        end
    end
    return df
end

## Load dataset
star = dataset("mlmRev", "star")

## Preprocessing
star_complete = complete_cases!(copy(star))
panel = encode_frame(star_complete)

## Training set

## Generate train/test split
# Source:
# http://blog.yhathq.com/posts/julia-neural-networks.html

#features = convert(Array{Float64}, DataArray(panel[[:,:,:,:,:,:]]))
#labels = convert(Array{Float64}, panel[:SES])

X = convert(Array{Float64}, DataArray(panel[[:Sch,:ClType,:HDeg,:Exp,:Trace,:Read,:Math,:SchType,:Eth]]))
y = convert(Array{Float64}, panel[:SES])

n = size(panel, 1)
is_train = shuffle([1:n] .> floor(n * .25))

X_train, X_test = X[is_train,:], X[!is_train,:]
y_train, y_test = y[is_train], y[!is_train]

println("Total number of observations: ",n)
println("Training set size: ", sum(is_train))
println("Test set size: ", sum(!is_train))

model = build_forest(labels, features, 4, 100, 0.5)
accuracy = nfoldCV_forest(labels, feats, 4, 100, 3, 0.5)
