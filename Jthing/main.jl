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

function run_isc(stim::Array{String,1}=["S1","S2","S3","S4"],pref::String="S")
  interval = 1:3620
  const means = [ prep(subj) for subj in 1:NSUBJ ]
  @debug("means done")
  covs = zeros(Float64, SIZE,div(NSUBJ*(NSUBJ-1),2))
  disps = zeros(Float64, (SIZE,NSUBJ))
  indexes = [get_indexes(i,stim) for i in 1:NSUBJ]
  for j in 1:size(indexes[1],1)
    local seq = [get_niis(i,indexes[i][j,1]:indexes[i][j,2]) for i in 1:NSUBJ]
    @debug("Got niis")
    cov_n!(covs,disps,seq, means,17)
    @debug("Done with : $j")
  end
  @debug("covs_n done")
  save("$pref|covs.jld", "data", covs)
  save("$pref|disps.jld", "data", disps)
  # niscores = reshape([score(corrs[i,:,:]) for i in 1:size(corrs,1)],SHAPE)
  niscores = ISC_res(covs,disps)
  @info("Scores done")
  cleared = map(x->isnan(x)?0:min(abs(x),1)*sign(x),niscores)
  ni = get_nii(1,1)
  res = NIfTI.NIVolume(ni.header, ni.extensions, reshape(cleared, SHAPE))
  niwrite("$pref|segmentation_result.nii",res)
  covs,niscores,disps
end

run_isc()
@debug("S done!")
run_isc(["V1","V2"],"V")
@debug("V done!")
run_isc(["S1","S2","S3","S4","V1","V2"],"S_V")
@debug("S and V done!")
