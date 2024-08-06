import stainNorm_Reinhard
import stainNorm_Vahadane
import stainNorm_Macenko
import cv2
import stain_utils
target_image_path = "/media/lzs/41e45ae4-ad45-41db-9bc2-b8299dd12de8/survival_prediction/code/survival_prediction/reference/12_44.png"
source_image_path = "/media/lzs/T7 Shield/patch/201503063_HE_3/2_90.png"
target_image = cv2.imread(target_image_path)
source_image = cv2.imread(source_image_path)

M = stainNorm_Macenko.Normalizer()
M.fit(target_image)
out_M = M.transform(source_image)

R = stainNorm_Reinhard.Normalizer()
R.fit(target_image)
out_R = R.transform(source_image)

V = stainNorm_Vahadane.Normalizer()
V.fit(target_image)
out_V = V.transform(source_image)
# n = stainNorm_Vahadane.Normalizer()
# out = n.hematoxylin(source_image)
# n = stainNorm_Vahadane.Normalizer()
# n.fit(target_image)
# stain_utils.show_colors(n.target_stains())
cv2.imwrite("/media/lzs/T7 Shield/result/zhejiang_3/2_90.png", source_image)
cv2.imwrite("/media/lzs/T7 Shield/result/zhejiang_3/2_90_1.png", out_M)
cv2.imwrite("/media/lzs/T7 Shield/result/zhejiang_3/2_90_2.png", out_R)
cv2.imwrite("/media/lzs/T7 Shield/result/zhejiang_3/2_90_3.png", out_V)