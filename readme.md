# Model self-driving
Mô hình sử dụng chuỗi gồm 5 ảnh liên tiếp để predict các hành động của việc điều khiển xe trong game bao gồm: s,w,a,d,wa,wd,sa,sd
Mô hình bảo gồm 3 phần chính.
- extract feature images thành các embedding, sử dụng efficient-b4
- positional encoder
- encoder transformer
- outputlayer
- predict 8 class 
# Model segmentation
Sử dụng mô hình sfsegment cho fps cao hơn khi sử dụng deeplab3plus cho bộ dữ liệu CityScapes
# Model object detection
Sử dụng mô hình yolov5 với pretrained yolov5s.pt sử dụng tensorrt engine yolov5s.engine để tăng tốc predict 
# Tracking
Sử dụng deepsort để tracking object kết hợp cùng yolov5
