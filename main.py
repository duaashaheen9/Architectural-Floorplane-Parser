import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from ultralytics import YOLO

# Target processing spatial dimension constant
IMG_SIZE = 512

# ================================================================
# 1. Context Segmentation Model Architecture
# ================================================================
def transformer_block(x, num_heads=8, ff_dim=1024, dropout_rate=0.1, survival_prob=0.9):
    """Deep Core Multi-Head Attention block with stochastic layer survival."""
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(x, x)
    attn = layers.Dropout(dropout_rate)(attn)
    surve = tf.cast(tf.random.uniform(()) < survival_prob, tf.float32)
    x = layers.LayerNormalization(epsilon=1e-6)(x + surve * attn)
    ff = layers.Dense(ff_dim, activation='relu')(x)
    ff = layers.Dense(x.shape[-1])(ff)
    ff = layers.Dropout(dropout_rate)(ff)
    return layers.LayerNormalization(epsilon=1e-6)(x + ff)

def build_full_model():
    """Constructs the hybrid ResNet50 + Transformer Semantic Segmentation model."""
    base = keras.applications.ResNet50(include_top=False, weights=None, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    
    # Isolate functional residual multi-scale skip layers
    skip1 = base.get_layer('conv2_block3_out').output
    skip2 = base.get_layer('conv3_block4_out').output
    skip3 = base.get_layer('conv4_block6_out').output
    cnn_out = base.get_layer('conv5_block3_out').output
    
    x = layers.Conv2D(512, 1, padding='same')(cnn_out)
    h, w, c = 16, 16, 512
    x = layers.Reshape((h*w, c))(x)
    
    pos_emb = layers.Embedding(h*w, c)(tf.range(h*w))
    x = x + pos_emb
    
    for i in range(4):
        x = transformer_block(x, num_heads=8, ff_dim=1024, dropout_rate=0.1, survival_prob=0.9 - i*0.1)
        
    transformer_features = layers.LayerNormalization(name='transformer_features')(x)
    x = layers.Reshape((h, w, c))(transformer_features)
    
    def decoder_block(x, filters, skip=None):
        x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
        if skip is not None:
            skip = layers.Conv2D(filters, 1, padding='same')(skip)
            x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        return x
        
    x = decoder_block(x, 256, skip3)
    x = decoder_block(x, 128, skip2)
    x = decoder_block(x, 64, skip1)
    
    x = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    
    x = layers.Conv2DTranspose(16, 2, strides=2, padding='same')(x)
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(x)
    
    seg_out = layers.Conv2D(1, 1, activation='sigmoid', name='segmentation_mask')(x)
    return keras.Model(inputs=base.input, outputs=[seg_out, transformer_features])

# Instantiate and build deep weight frameworks
print("Initializing deep vision networks...")
seg_model = build_full_model()
seg_model.load_weights('floorplan_segmentation_final.keras')
yolo_model = YOLO('best.pt')
print("Model states loaded successfully.")

# ================================================================
# 2. GrabCut Probability-Guided Refinement
# ================================================================
def extract_room_guided(img_resized, prob_map, x1, y1, x2, y2):
    """Refines localized YOLO coordinates using mask probability constraints as a prior."""
    roi_img = img_resized[y1:y2, x1:x2]
    roi_prob = prob_map[y1:y2, x1:x2]
    h, w = roi_prob.shape
    if h == 0 or w == 0:
        return None
        
    mask = np.zeros(roi_prob.shape, np.uint8)
    mask[roi_prob > 0.5] = cv2.GC_PR_FGD
    mask[roi_prob <= 0.5] = cv2.GC_PR_BGD
    
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (0, 0, w, h)
    
    cv2.grabCut(roi_img, mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_MASK)
    final_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    
    kernel = np.ones((3, 3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    cnt_local = max(contours, key=cv2.contourArea)
    cnt_full = cnt_local + np.array([x1, y1])
    return cnt_full

# ================================================================
# 3. Vectorized Core Hybrid Execution Pipeline
# ================================================================
def predict_floorplan_hybrid(image_path, seg_model, yolo_model, output_dir='final_result',
                             yolo_conf=0.15, iou_thresh=0.25):
    """Parses blueprint matrices into clean coordinate topologies and a single visualization plot."""
    os.makedirs(output_dir, exist_ok=True)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Image not discovered at path: {image_path}")
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

    # Bounding Box Proposal Extractor (YOLO Stage)
    results = yolo_model(img_resized, conf=yolo_conf, iou=iou_thresh)
    raw_boxes, confs = [], []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:  # Omit background wall anchors
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            if w > IMG_SIZE * 0.65 or h > IMG_SIZE * 0.65: continue
            if w < 10 or h < 10: continue
            raw_boxes.append([x1, y1, w, h])
            confs.append(float(box.conf[0]))

    final_boxes = []
    if raw_boxes:
        indices = cv2.dnn.NMSBoxes(raw_boxes, confs, score_threshold=yolo_conf, nms_threshold=0.3)
        if len(indices) > 0:
            for i in indices.flatten():
                rb = raw_boxes[i]
                final_boxes.append((rb[0], rb[1], rb[0] + rb[2], rb[1] + rb[3]))

    if not final_boxes:
        print("⚠️ Processing discontinued: No valid structural regions found.")
        return {'is_valid_blueprint': False, 'rooms': []}

    # Context Feature Predictor (Transformer Stage)
    img_in = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)
    seg_pred, feat_pred = seg_model.predict(img_in, verbose=0)
    prob_map = seg_pred[0, :, :, 0]

    rooms = []
    for idx, (x1, y1, x2, y2) in enumerate(final_boxes):
        cnt_full = extract_room_guided(img_resized, prob_map, x1, y1, x2, y2)
        if cnt_full is None:
            continue

        # Polygon Vector Corner Tracking Approximation
        epsilon = 0.02 * cv2.arcLength(cnt_full, True)
        approx = cv2.approxPolyDP(cnt_full, epsilon, True)
        if len(approx) == 4:
            corners = approx.reshape(4, 2).tolist()
        else:
            rect = cv2.minAreaRect(cnt_full)
            corners = cv2.boxPoints(rect).astype(int).tolist()

        # Spatial Mass Center Identification
        M = cv2.moments(cnt_full)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Correlating latent transformer features maps
        feat_x = min(max(int(cx / 32), 0), 15)
        feat_y = min(max(int(cy / 32), 0), 15)
        feat_vec = feat_pred[0, feat_y * 16 + feat_x, :].tolist()

        rooms.append({
            'room_id': idx + 1,
            'center': {'x': cx, 'y': cy},
            'corners': [{'x': int(pt[0]), 'y': int(pt[1])} for pt in corners],
            'area_pixels': int(cv2.contourArea(cnt_full)),
            'transformer_features': feat_vec,
            'full_contour': cnt_full
        })

    # Render Unified Labeled Bounding Contours Layout Plot
    overlay = img_resized.copy()
    for room in rooms:
        cv2.drawContours(overlay, [room['full_contour']], -1, (0, 255, 0), 2)
        cv2.circle(overlay, (room['center']['x'], room['center']['y']), 4, (255, 0, 0), -1)
        cv2.putText(overlay, str(room['room_id']),
                    (room['center']['x'] - 8, room['center']['y'] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
    labeled_path = os.path.join(output_dir, 'labeled_floorplan.png')
    cv2.imwrite(labeled_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Structured Data Export
    json_rooms = []
    for room in rooms:
        room_copy = {k: v for k, v in room.items() if k != 'full_contour'}
        json_rooms.append(room_copy)

    with open(os.path.join(output_dir, 'rooms_data.json'), 'w') as f:
        json.dump({'is_valid_blueprint': True, 'rooms': json_rooms}, f, indent=4)

    print(f"✅ Framework execution finished: {len(json_rooms)} deep-guided topologies generated.")
    return {'is_valid_blueprint': True, 'rooms': json_rooms}

# ================================================================
# 4. Pipeline Execution Trigger
# ================================================================
if __name__ == '__main__':
    test_img = 'sample_floorplan.jpg'
    res = predict_floorplan_hybrid(test_img, seg_model, yolo_model, output_dir='final_result')
