module Vis
push!(LOAD_PATH,".")

using ImageView, GtkReactive, Colors

export vis, vis_col

function vis(mri_, mask, thresh=0.1)
  mri = Gray.(copy(mri_)/ maximum(mri_))
  mriseg = RGBA.(mri)
  mriseg[mask .> thresh] += RGBA(0.,0.,1.,0.5)
  zr, slicedata = roi(mri, (1,2))
  gd = imshow_gui((200, 200), slicedata, (1,2))
  imshow(gd["frame"][1,1], gd["canvas"][1,1], mri, nothing, zr, slicedata)
  imshow(gd["frame"][1,2], gd["canvas"][1,2], mriseg, nothing, zr, slicedata)
  showall(gd["window"])
end

function vis_col(mri_ ,mask,thresh = 0.1)
  mri = Gray.(copy(mri_)/ maximum(mri_))
  mriseg = RGB.(mri)
  cm = colormap("Oranges", 255)
  mriseg[mask .> thresh] = cm[Int.(ceil.(255*mask[mask .> thresh]))]
  zr, slicedata = roi(mri, (1,2))
  gd = imshow_gui((200, 200), slicedata, (1,2))
  imshow(gd["frame"][1,1], gd["canvas"][1,1], mri, nothing, zr, slicedata)
  imshow(gd["frame"][1,2], gd["canvas"][1,2], mriseg, nothing, zr, slicedata)
  showall(gd["window"])
end

end
