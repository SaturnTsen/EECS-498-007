FPN Feature Pyramid Network
Classification: { levels: cls_out = (B, H*W, NUM_CLASSES) }
Boxreg:         { levels: box_out = (B, H*W, 4) }
Centerness:     { levels: ctr_out = (B, H*W, 1) }

get_fpn_location_coords：(meshgrid)
    shape_per_fpn_level strides_per_fpn_level dtype, device
--> locations_per_fpn_level { levels: (H*W, 2) giving (xc, yc) 感受野中心} 

fcos_match_locations_to_gt:
    locations_per_fpn_level
    strides_per_fpn_level
    gt_boxes
--> matched_boxes_per_fpn_level { levels: (H*W, 5) }
    (H*W,  x  y  x  y  c) or
    (H*W, -1 -1 -1 -1 -1)

fcos_get_deltas_from_locations:
    locations,         gt_boxes,      stride
    (H*W, 2) (xc, yc)  (H*W, 4 or 5)
--> (H*W, 4)

fcos_apply_deltas_to_locations
    _deltas, input_locations, stride
--> (H*W, 4) xyxy

fcos_make_centerness_targets
    deltas
--> centerness
