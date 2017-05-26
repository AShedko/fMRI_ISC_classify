push!(LOAD_PATH, ".")

using ISCLib

using HDF5, JLD, NIfTI, Logging
@Logging.configure(level=DEBUG)

function score(m::Array{Float64,1}, bound=0.4)
  dim = size(m,1)
  score = sum(m)/(dim^2)
  if score>=bound
    return score
  else
    return 0
  end
end

function separate(interval::AbstractArray, lim =100)
    l = length(interval)
  if l>lim
    nchunks = Int(ceil(l/lim))
    [interval[(j-1)*lim+1:((j)*lim<l?(j)*lim:end)]for j in 1:nchunks]
  else
    [interval]
  end
end

function main()
  interval = 1:3620
  const means = [ prep(subj) for subj in 1:NSUBJ ]
  @debug("means done")
  covs = zeros(Float64, SIZE,NSUBJ*div((NSUBJ-1),2))
  disps = zeros(Float64, (SIZE,NSUBJ))
  indexes = [get_indexes(i) for i in 1:NSUBJ]
  for j in 1:length(indexes[1])
    local seq = [get_niis(i,indexes[j][1]:indexes[j][2]) for i in 1:NSUBJ]
    @debug("Got niis")
    cov_n!(covs,disps,seq, means,indexes[j][2]-indexes[j][1])
    @debug("Done with : $j")
  end
  @debug("covs_n done")
  save("covs.jld", "data", covs)
  save("disps.jld", "data", disps)
  # niscores = reshape([score(corrs[i,:,:]) for i in 1:size(corrs,1)],SHAPE)
  niscores = ISC_res(covs,disps)
  @info("Scores done")
  cleared = map(x->isnan(x)?0:min(abs(x),1)*sign(x),niscores)
  ni = get_nii(1,1)
  res = NIfTI.NIVolume(ni.header, ni.extensions, reshape(cleared, SHAPE))
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
