
import os, json, cv2, numpy as np, random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pycocotools.coco import COCO
from ultralytics import YOLO

IMG_SIZE = 512
BATCH_SIZE = 4
NEG_RATIO = 0.3
MAX_NEG_SAMPLES = 2000

CUBICASA_TRAIN_JSON = '/kaggle/input/datasets/qmarva/cubicasa5k/cubicasa5k_coco/train_coco_pt.json'
CUBICASA_VAL_JSON   = '/kaggle/input/datasets/qmarva/cubicasa5k/cubicasa5k_coco/val_coco_pt.json'
IMAGES_ROOT         = '/kaggle/input/datasets/qmarva/cubicasa5k/cubicasa5k/cubicasa5k/high_quality_architectural'
NEG_IMG_DIR         = '/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/val2017'

TRAIN_MASK_DIR = '/kaggle/working/train_masks_real'
VAL_MASK_DIR   = '/kaggle/working/val_masks_real'
os.makedirs(TRAIN_MASK_DIR, exist_ok=True)
os.makedirs(VAL_MASK_DIR, exist_ok=True)

def get_rel_path(abs_path):
    return abs_path.split('high_quality_architectural/')[-1]

def create_masks_from_coco(json_path, mask_dir):
    coco = COCO(json_path)
    non_room_names = {'wall', 'door', 'window', 'column', 'railing', 'opening', 'stairs'}
    room_cat_ids = [c['id'] for c in coco.loadCats(coco.getCatIds()) if c['name'] not in non_room_names]
    print(f"Room categories: {[coco.loadCats(i)[0]['name'] for i in room_cat_ids]}")
    for img_id in coco.getImgIds():
        info = coco.loadImgs(img_id)[0]
        rel = get_rel_path(info['file_name'])
        img_full = os.path.join(IMAGES_ROOT, rel)
        if not os.path.exists(img_full): continue
        img = cv2.imread(img_full)
        if img is None: continue
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=room_cat_ids)
        for ann in coco.loadAnns(ann_ids):
            if 'segmentation' in ann and ann['segmentation']:
                try:
                    m = coco.annToMask(ann) * 255
                    mask = cv2.bitwise_or(mask, m.astype(np.uint8))
                except: pass
            else:
                bbox = ann['bbox']
                x,y,w,h = map(int, bbox)
                cv2.rectangle(mask, (x,y), (x+w,y+h), 255, -1)
        rel_no_ext = os.path.splitext(rel)[0]
        mask_rel = rel_no_ext + '.png'
        sub = os.path.join(mask_dir, os.path.dirname(mask_rel))
        os.makedirs(sub, exist_ok=True)
        cv2.imwrite(os.path.join(mask_dir, mask_rel), mask)
    print(f"Masks created in {mask_dir}")

def build_file_lists(json_path, mask_dir):
    coco = COCO(json_path)
    imgs, masks = [], []
    for img_id in coco.getImgIds():
        info = coco.loadImgs(img_id)[0]
        rel = get_rel_path(info['file_name'])
        img_full = os.path.join(IMAGES_ROOT, rel)
        rel_no_ext = os.path.splitext(rel)[0]
        mask_full = os.path.join(mask_dir, rel_no_ext + '.png')
        if os.path.exists(img_full) and os.path.exists(mask_full):
            imgs.append(img_full)
            masks.append(mask_full)
    return imgs, masks

def load_positive(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [IMG_SIZE, IMG_SIZE], method='nearest')
    mask = tf.cast(mask > 127, tf.float32)
    mask = tf.squeeze(mask, axis=-1)
    return img, mask

def augment_positive(img, mask):
    mask = tf.expand_dims(mask, -1)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k)
    mask = tf.image.rot90(mask, k)
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    img = tf.clip_by_value(img, 0.0, 1.0)
    mask = tf.squeeze(mask, axis=-1)
    return img, mask

def load_negative(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    mask = tf.zeros([IMG_SIZE, IMG_SIZE], dtype=tf.float32)
    return img, mask

def augment_negative(img, mask):
    mask = tf.expand_dims(mask, -1)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.clip_by_value(img, 0.0, 1.0)
    mask = tf.squeeze(mask, axis=-1)
    return img, mask

def build_combined_train_ds(pos_imgs, pos_masks, neg_dir, batch_size, neg_ratio=NEG_RATIO):
    pos_bs = max(1, int(batch_size * (1 - neg_ratio)))
    neg_bs = batch_size - pos_bs
    combined_pos = list(zip(pos_imgs, pos_masks))
    np.random.shuffle(combined_pos)
    pos_imgs_s, pos_masks_s = zip(*combined_pos)
    pos_ds = tf.data.Dataset.from_tensor_slices((list(pos_imgs_s), list(pos_masks_s)))
    pos_ds = pos_ds.map(load_positive, num_parallel_calls=tf.data.AUTOTUNE)
    pos_ds = pos_ds.map(augment_positive, num_parallel_calls=tf.data.AUTOTUNE)
    pos_ds = pos_ds.batch(pos_bs)
    neg_files = [os.path.join(neg_dir, f) for f in os.listdir(neg_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))][:MAX_NEG_SAMPLES]
    np.random.shuffle(neg_files)
    neg_ds = tf.data.Dataset.from_tensor_slices(neg_files)
    neg_ds = neg_ds.map(load_negative, num_parallel_calls=tf.data.AUTOTUNE)
    neg_ds = neg_ds.map(augment_negative, num_parallel_calls=tf.data.AUTOTUNE)
    neg_ds = neg_ds.batch(neg_bs)
    combined = tf.data.Dataset.zip((pos_ds, neg_ds))
    combined = combined.map(lambda p, n: (tf.concat([p[0], n[0]], axis=0), tf.concat([p[1], n[1]], axis=0)),
                            num_parallel_calls=tf.data.AUTOTUNE)
    return combined.shuffle(100).prefetch(tf.data.AUTOTUNE)

def transformer_block(x, num_heads=8, ff_dim=1024, dropout_rate=0.1, survival_prob=0.9):
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(x, x)
    attn = layers.Dropout(dropout_rate)(attn)
    survive = tf.cast(tf.random.uniform(()) < survival_prob, tf.float32)
    x = layers.LayerNormalization(epsilon=1e-6)(x + survive * attn)
    ff = layers.Dense(ff_dim, activation='relu')(x)
    ff = layers.Dense(x.shape[-1])(ff)
    ff = layers.Dropout(dropout_rate)(ff)
    return layers.LayerNormalization(epsilon=1e-6)(x + ff)

def build_full_model():
    base = keras.applications.ResNet50(include_top=False, weights=None, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
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
        x = transformer_block(x, num_heads=8, ff_dim=1024, dropout_rate=0.1, survival_prob=0.9-i*0.1)
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

def dice_loss(y_true, y_pred):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return 1.0 - (2.*intersection+1e-7)/(union+1e-7)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)
    return 0.5*bce + 0.5*dice_loss(y_true, y_pred)

class SegmentationIoU(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name='iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.intersection = self.add_weight(name='inter', initializer='zeros')
        self.union = self.add_weight(name='union', initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]) > self.threshold, tf.float32)
        self.intersection.assign_add(tf.reduce_sum(y_true * y_pred))
        self.union.assign_add(tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - tf.reduce_sum(y_true * y_pred) + 1e-7)
    def result(self): return self.intersection / self.union
    def reset_state(self): self.intersection.assign(0.0); self.union.assign(0.0)

def compile_model(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=combined_loss, metrics=[SegmentationIoU(threshold=0.5, name='iou')])

create_masks_from_coco(CUBICASA_TRAIN_JSON, TRAIN_MASK_DIR)
create_masks_from_coco(CUBICASA_VAL_JSON, VAL_MASK_DIR)

train_imgs, train_masks = build_file_lists(CUBICASA_TRAIN_JSON, TRAIN_MASK_DIR)
val_imgs, val_masks = build_file_lists(CUBICASA_VAL_JSON, VAL_MASK_DIR)
print(f"Training samples: {len(train_imgs)}, Validation: {len(val_imgs)}")

train_ds = build_combined_train_ds(train_imgs, train_masks, NEG_IMG_DIR, BATCH_SIZE)
val_ds = tf.data.Dataset.from_tensor_slices((val_imgs, val_masks))
val_ds = val_ds.map(load_positive, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

full_model = build_full_model()
full_model.load_weights('floorplan_segmentation_final.keras')
training_model = keras.Model(inputs=full_model.input, outputs=full_model.output[0])
compile_model(training_model, lr=1e-4)

callbacks = [
    keras.callbacks.ModelCheckpoint('best_finetuned.keras', save_best_only=True, monitor='val_loss'),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

print("Phase 1: frozen backbone")
training_model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)
full_model.layers[0].trainable = True
compile_model(training_model, lr=1e-5)
print("Phase 2: full fine‑tuning")
training_model.fit(train_ds, validation_data=val_ds, epochs=50, initial_epoch=20, callbacks=callbacks)
full_model.save('floorplan_segmentation_improved.keras')
print("Improved model saved.")

yolo_model = YOLO('best.pt')

def extract_rooms(pred_mask, transformer_features, original_image, output_dir='rooms_out'):
    os.makedirs(output_dir, exist_ok=True)
    bin_mask = (pred_mask > 0.5).astype(np.uint8)*255
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(bin_mask, kernel, iterations=1)
    dist = cv2.distanceTransform(eroded, cv2.DIST_L2, 3)
    _, sure_fg = cv2.threshold(dist, 0.4*dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_fg)
    rooms = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 200: continue
        room_mask = np.uint8(labels==i)*255
        contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        cnt = max(contours, key=cv2.contourArea)
        epsilon = 0.02*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx)==4: corners = approx.reshape(4,2).tolist()
        else: rect = cv2.minAreaRect(cnt); corners = cv2.boxPoints(rect).astype(int).tolist()
        M = cv2.moments(room_mask)
        if M['m00']!=0: cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        else: cx, cy = int(centroids[i][0]), int(centroids[i][1])
        x,y,w,h = cv2.boundingRect(cnt)
        crop = original_image[y:y+h, x:x+w]
        crop_path = os.path.join(output_dir, f'room_{i}.png')
        cv2.imwrite(crop_path, crop)
        feat_x = min(max(int(cx/32),0),15); feat_y = min(max(int(cy/32),0),15)
        feat_vec = transformer_features[0, feat_y*16+feat_x, :].tolist()
        rooms.append({'room_id':i,'center':{'x':cx,'y':cy},'corners':[{'x':int(p[0]),'y':int(p[1])} for p in corners],
                      'cropped_image_path':crop_path,'area_pixels':area,'transformer_features':feat_vec,'full_contour':cnt})
    return rooms

def draw_labeled_floorplan(original_image, pred_mask, rooms, alpha=0.35):
    overlay = original_image.copy()
    colored = np.zeros_like(original_image)
    colored[:,:,0] = (pred_mask*255).astype(np.uint8)
    cv2.addWeighted(colored, alpha, overlay, 1-alpha, 0, overlay)
    for room in rooms:
        cv2.drawContours(overlay, [room['full_contour']], -1, (0,255,0), 2)
        cv2.circle(overlay, (room['center']['x'], room['center']['y']), 5, (255,0,0), -1)
        cv2.putText(overlay, str(room['room_id']), (room['center']['x']-12, room['center']['y']-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return overlay

def predict_floorplan_combined(image_path, seg_model, yolo_model, output_dir='combined_out'):
    os.makedirs(output_dir, exist_ok=True)
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    results = yolo_model(img, conf=0.3)
    boxes = [tuple(map(int, b.xyxy[0])) for r in results for b in r.boxes if int(b.cls[0])==1]
    if not boxes: return {'is_valid_blueprint':False,'rooms':[]}
    img_in = np.expand_dims(img.astype(np.float32)/255.0, 0)
    seg_pred, feat_pred = seg_model.predict(img_in, verbose=0)
    prob_map = seg_pred[0,:,:,0]
    rooms = []
    for idx,(x1,y1,x2,y2) in enumerate(boxes):
        pad = 5
        x1p, y1p = max(0,x1-pad), max(0,y1-pad)
        x2p, y2p = min(IMG_SIZE,x2+pad), min(IMG_SIZE,y2+pad)
        crop_prob = prob_map[y1p:y2p, x1p:x2p]
        blur = cv2.GaussianBlur(crop_prob, (5,5), 0)
        _, local_mask = cv2.threshold((blur*255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        cnt_local = max(contours, key=cv2.contourArea)
        cnt_full = cnt_local + np.array([x1p, y1p])
        epsilon = 0.02*cv2.arcLength(cnt_full, True)
        approx = cv2.approxPolyDP(cnt_full, epsilon, True)
        if len(approx)==4: corners = approx.reshape(4,2).tolist()
        else: rect = cv2.minAreaRect(cnt_full); corners = cv2.boxPoints(rect).astype(int).tolist()
        M = cv2.moments(local_mask)
        if M['m00']!=0: cx, cy = int(M['m10']/M['m00'])+x1p, int(M['m01']/M['m00'])+y1p
        else: cx, cy = (x1+x2)//2, (y1+y2)//2
        bx, by, bw, bh = cv2.boundingRect(cnt_full)
        crop_img = img[by:by+bh, bx:bx+bw]
        crop_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        cv2.drawContours(crop_mask, [cnt_full], -1, 255, -1)
        crop_mask = crop_mask[by:by+bh, bx:bx+bw]
        bgra = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGRA)
        bgra[:,:,3] = crop_mask
        crop_path = os.path.join(output_dir, f'room_{idx+1}.png')
        cv2.imwrite(crop_path, bgra)
        feat_x = min(max(int(cx/32),0),15); feat_y = min(max(int(cy/32),0),15)
        feat_vec = feat_pred[0, feat_y*16+feat_x, :].tolist()
        rooms.append({'room_id':idx+1,'center':{'x':cx,'y':cy},'corners':[{'x':int(p[0]),'y':int(p[1])} for p in corners],
                      'cropped_image_path':crop_path,'area_pixels':int(cv2.contourArea(cnt_full)),'transformer_features':feat_vec,
                      'full_contour':cnt_full})
    overlay = img.copy()
    colored = np.zeros_like(img)
    colored[:,:,0] = (prob_map>0.5).astype(np.uint8)*255
    cv2.addWeighted(colored, 0.3, overlay, 0.7, 0, overlay)
    for room in rooms:
        cv2.drawContours(overlay, [room['full_contour']], -1, (0,255,0), 2)
        cv2.circle(overlay, (room['center']['x'], room['center']['y']), 5, (255,0,0), -1)
        cv2.putText(overlay, str(room['room_id']), (room['center']['x']-12, room['center']['y']-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.imwrite(os.path.join(output_dir,'labeled_floorplan.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    for room in rooms: del room['full_contour']
    with open(os.path.join(output_dir,'rooms_data.json'),'w') as f:
        json.dump({'is_valid_blueprint':True,'rooms':rooms,'labeled_image_path':os.path.join(output_dir,'labeled_floorplan.png')}, f, indent=4)
    print(f"Done – {len(rooms)} rooms.")