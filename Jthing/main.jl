push!(LOAD_PATH, ".")
using ISCLib
import ISCLib
using HDF5, JLD, NIfTI

function score(m::Array{Float64,2}, bound=0.4)
  dim = size(m,1)
  score = sum(m)/(dim^2)
  if score>=bound
    return score
  else
    return 0
  end
end

function main()
  interval = 1:LEN
  means = [ prep(subj) for subj in 1:NSUBJ ]
  corrs,disps  = cor_n(means, interval)
  save("corrs.jld", "data", corrs)
  save("disps.jld", "data", disps)
  niscores = reshape([score(corrs[i,:,:]) for i in 1:size(corrs,1)],SHAPE)
  res = NIfTI.NIVolume(means[1].header, subjects[1].extensions, niscores)
  niwrite("segmentation_result.nii",res)
  corrs,niscores,disps
end

@time CS,res,DS = main()

function rema_!(shadow::Float64,variable::Float64,decay::Float64= 0.3)
    shadow -= (1 - decay) * (shadow - variable)
end

function smooth(x,decay=0.9)
  res = copy(x)
  st = x[1]
  res[1] = st
  i = 2
  for el in x[2:end]
    st = rema_!(st,el,decay)
    res[i] = st
    i+=1
  end
  res
end
