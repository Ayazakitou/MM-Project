# image_retrieval_engine.py
import cv2 as cv
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from glob import glob
import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from skimage import feature
import pickle

database_dir = "image.orig"
query_dir = "image.query"
feature_cache_file = "features_cache.pkl"

class MediaRetrieval:
    def __init__(self):
        print("Loading deep learning models...")
        
        self.vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')

        self.database_features = {}
        
    def image_read(self, img_path):
        try:
            img = cv.imread(img_path)
            if img is None:
                print(f"No image read {img_path}")
                return None
            return img
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            return None
        
    def ensure_cache(self):
        if not os.path.exists(feature_cache_file):
            return False
        elif os.path.getsize(feature_cache_file) == 0:
            return False
        
        return True
    
    def extract_deep(self, img):
        try:
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            image_resized = cv.resize(img_rgb, (224, 224))
            img_prep = preprocess_input(image_resized)
            img_expand = np.expand_dims(img_prep, axis=0)

            v_features = self.vgg_model.predict(img_expand, verbose=0)
            v_features = v_features.flatten()

            norm_v = np.linalg.norm(v_features)
            if norm_v > 0:
                v_features /= norm_v
            return v_features
        
        except Exception as e:
            print(f"Deep feature extraction Error: {e}")
            return np.zeros(4096)
    
    def extract_moments(self, img):
        try:
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            
            h_mean = np.mean(hsv[:,:,0])
            h_std = np.std(hsv[:,:,0])
            s_mean = np.mean(hsv[:,:,1])
            s_std = np.std(hsv[:,:,1])
            v_mean = np.mean(hsv[:,:,2])
            v_std = np.std(hsv[:,:,2])

            features = np.array([h_mean, h_std, s_mean, s_std, v_mean, v_std])
            features = np.clip(features, 0, 255)
            
            return features
        except Exception as e:
            print(f"Color moments extraction Error: {e}")
            return np.zeros(6)
    
    def extract_texture(self, img):
        try:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
            lbp = feature.local_binary_pattern(gray, P=8, R=1, method="uniform")
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            lbp_hist = lbp_hist.astype(np.float32)
            if lbp_hist.sum() > 0:
                lbp_hist /= lbp_hist.sum()
            
            texture_f = self.compute_texture(gray)
            
            return np.concatenate([lbp_hist[:50], texture_f])
        except Exception as e:
            print(f"Texture extraction Error: {e}")
            return np.zeros(60)

    def compute_texture(self, gray_img):
        try:

            edges = cv.Canny(gray_img, 100, 200)
            edge_density = np.sum(edges > 0)
            edge_density /= edges.size
            
            sobelx = cv.Sobel(gray_img, cv.CV_64F, 1, 0, ksize=3)
            sobely = cv.Sobel(gray_img, cv.CV_64F, 0, 1, ksize=3)
            
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            features = [
                edge_density,
                *(np.mean(gradient_magnitude), np.std(gradient_magnitude)),
            ]
            
            hist, _ = np.histogram(gray_img, bins=256, range=(0, 256))
            hist = hist.astype(np.float32)
            hist /= hist.sum()
            entropy = 0.0

            for x in hist:
                if x > 0:
                    x *= np.log2(x)
                    entropy -= x

            features.append(entropy)

            for x in features:
                if np.isnan(x) or np.isinf(x):
                    x = 0.0
            
            return np.array(features)
        except Exception as e:
            print(f"Compute texture feature Error: {e}")
            return np.zeros(3)
    
    def extract_shape(self, img):
        try:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
            binary = cv.adaptiveThreshold(
                gray,
                255,  
                cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv.THRESH_BINARY, 11, 2
                )
            
            contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            contour_f = []
            for cnt in contours[:3]:
                if len(cnt) >= 5:
                    area = cv.contourArea(cnt)
                    perimeter = cv.arcLength(cnt, True)
                    
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                    else:
                        circularity = 0
                    
                    x, y, w, h = cv.boundingRect(cnt)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    
                    contour_f.extend([area, perimeter, circularity, aspect_ratio])
            
            while len(contour_f) < 12:
                contour_f.append(0.0)
            
            return np.array(contour_f[:12])
        except Exception as e:
            print(f"Shape feature extraction Error: {e}")
            return np.zeros(12)
    
    def combine_features(self, img):
        try:
            deep_f = self.extract_deep(img)
            
            color_f = self.extract_moments(img)
            texture_f = self.extract_texture(img)
            shape_f = self.extract_shape(img)
            
            features = np.concatenate([color_f, texture_f, shape_f])

            norm_v1 = np.linalg.norm(features)
            if norm_v1 > 0:
                features = features / norm_v1
            
            alpha = 0.7
            deep_f *= alpha
            features *= (1 - alpha)
            combined_f = np.concatenate([deep_f, features])
            
            norm_v2 = np.linalg.norm(combined_f)
            if norm_v2 > 0:
                combined_f = combined_f / norm_v2
            return combined_f.astype(np.float32)
            
        except Exception as e:
            print(f"Error building feature vector: {e}")
            return np.zeros(4096 + 6 + 60 + 12, dtype=np.float32)
    
    def compute_similarity(self, f1, f2):
        try:
            dot_product = np.dot(f1, f2)
            norm_v1 = np.linalg.norm(f1)
            norm_v2 = np.linalg.norm(f2)
            
            if norm_v1 > 0 and norm_v2 > 0:
                similarity = dot_product / (norm_v1 * norm_v2)
                similarity = max(0.0, min(1.0, similarity))
            else:
                similarity = 0.0
                
            return similarity
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.0
    
    def load_or_extract_f(self, image_paths):
        if os.path.exists(feature_cache_file) and self.ensure_cache():
            print("Loading cached features...")
            try:
                with open(feature_cache_file, 'rb') as features:
                    self.database_features = pickle.load(features)
                print(f"Loaded {len(self.database_features)} features from cache.")
                return True
            except Exception as e:
                print(f"Error loading cache: {e}. Re-extracting features...")

        print("Extracting features from database...")
        successful = 0
        for i, img_path in enumerate(image_paths):
            try:
                img = self.image_read(img_path)
                if img is not None:
                    features = self.combine_features(img)
                    self.database_features[img_path] = features
                    successful += 1
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i+1}/{len(image_paths)} images...")
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        try:
            with open(feature_cache_file, 'wb') as features:
                pickle.dump(self.database_features, features)
            print(f"Features cached. Successfully processed {successful}/{len(image_paths)} images.")
            return True
        except Exception as e:
            print(f"Error caching features: {e}")
    
    def search_similar(self, query_path, top_k=10):
        query_img = self.image_read(query_path)
        if query_img is None:
            return []
        
        query_f = self.combine_features(query_img)
        
        similarities = []
        for img_path, db_f in self.database_features.items():
            similarity = self.compute_similarity(query_f, db_f)
            similarities.append((img_path, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def clear_cache(self):
        if os.path.exists(feature_cache_file):
            os.remove(feature_cache_file)
            print("Feature cache cleared.")