push!(LOAD_PATH, ".")

using ISCLib

using HDF5, JLD, NIfTI, Logging
Logging.configure(level=DEBUG)

function process!(v::Array{Float64,1})
  map!(x-> isnan(x)?0:x, v ,v) # no nan`s
  map!(x-> x<0.1? 0: x, v, v) # no insignificant values-
  v
end

"""
rb - how many rows should be dropped from the back
"""
function run_isc(stim::Array{String,1}=["S1","S2"],pref::String="S",rb::Int=1)
  const means = [ get_mean(subj) for subj in 1:NSUBJ ]
  debug("means done")
  covs = zeros(Float32, SIZE,div(NSUBJ*(NSUBJ-1),2))
  disps = zeros(Float32, (SIZE,NSUBJ))
  indexes = [get_indexes(i,stim,rb) for i in 1:NSUBJ]
  for j in 1:size(indexes[1],1)
    local seq = [get_niis(i,indexes[i][j,1]:indexes[i][j,2]) for i in 1:NSUBJ]
    debug("Got niis")
    cov_n!(covs,disps,seq, means, 16) # 16 - 6 sec before 2 sec after => 9sec = 16 "frames"
    debug("Done with : $j")
  end
  debug("covs_n done")
  h5open("out/$pref|covs.h5","w") do file
    write(file, "data", covs)
  end
  h5open("out/$pref|disps.h5","w") do file
    write(file, "data", disps/(16*length(indexes)))
  end
  niscores = ISC_res(covs,disps)
  info("Scores done")

  res = process!(niscores)
  ni = get_nii(1,1)
  res = NIfTI.NIVolume(ni.header, ni.extensions, reshape(res, SHAPE)) # Cheap, as reshape returns a view
  niwrite("out/$pref|segmentation_result.nii",res)
  niscores
end



run_isc()
debug("S done!")
run_isc(["V1","V2","V3","V4"],"V")
debug("V done!")
run_isc(["S1","S2","V1","V2","V3","V4"],"S_V",2)
debug("S and V done!")
