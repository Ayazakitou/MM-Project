import cv2 as cv
import numpy as np
from glob import glob
import os
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import pickle

database_dir = "image.orig"
query_dir = "image.query"
feature_cache_file = "features_cache.pkl"

class MediaRetrieval:
    def __init__(self):
        print("Loading deep learning models...")
        # Pretrained models
        self.vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
        self.resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

        # Feature cache
        self.database_features = {}
        
    def image_read(self, img_path):
        try:
            img = cv.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                return None
            # Ensure 3 channels
            if len(img.shape) == 2:
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGR) 
            elif img.shape[2] == 4:
                img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
            return img
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            return None
    
    def extract_deep(self, img):
        try:
            # Preprocess image for models
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img_resized = cv.resize(img_rgb, (224, 224))
            img_array = np.expand_dims(img_resized, axis=0)
            img_preprocessed = preprocess_input(img_array.astype(np.float32))
            
            # VGG16
            vgg_features = self.vgg_model.predict(img_preprocessed, verbose=0)
            
            # ResNet50
            resnet_features = self.resnet_model.predict(img_preprocessed, verbose=0)
            
            # Combining
            combined_features = np.concatenate([vgg_features.flatten(), resnet_features.flatten()])
            
            # L2 normalization
            norm = np.linalg.norm(combined_features)
            if norm > 0:
                combined_features = combined_features / norm
            
            return combined_features
        except Exception as e:
            print(f"Deep feature extraction Error: {e}")
            return np.zeros(4096)
    
    def extract_moments(self, img):
        try:
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            
            h_mean, h_std = np.mean(hsv[:,:,0]), np.std(hsv[:,:,0])
            s_mean, s_std = np.mean(hsv[:,:,1]), np.std(hsv[:,:,1])
            v_mean, v_std = np.mean(hsv[:,:,2]), np.std(hsv[:,:,2])
            
            features = np.array([h_mean, h_std, s_mean, s_std, v_mean, v_std])
            features = np.clip(features, 0, 255)
            
            return features
        except Exception as e:
            print(f"Color moments extraction Error: {e}")
            return np.zeros(6)
    
    def extract_texture(self, img):
        try:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
            lbp = self.to_binaryP(gray)
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
            lbp_hist = lbp_hist.astype(np.float32)
            if lbp_hist.sum() > 0:
                lbp_hist = lbp_hist / lbp_hist.sum()
            
            texture_features = self.compute_texture(gray)
            
            return np.concatenate([lbp_hist[:50], texture_features])
        except Exception as e:
            print(f"Texture extraction Error: {e}")
            return np.zeros(60)
    
    def to_binaryP(self, img, radius=1, neighbors=8):
        height, width = img.shape
        lbp = np.zeros((height-2*radius, width-2*radius), dtype=np.uint8)
        
        for i in range(radius, height-radius):
            for j in range(radius, width-radius):
                center = img[i, j]
                binary_pattern = 0
                for k in range(neighbors):
                    angle = 2 * np.pi * k / neighbors
                    x = i + int(radius * np.cos(angle))
                    y = j - int(radius * np.sin(angle))
                    
                    x = max(0, min(x, height-1))
                    y = max(0, min(y, width-1))
                    
                    if img[x, y] >= center:
                        binary_pattern |= (1 << (neighbors - 1 - k))
                
                lbp[i-radius, j-radius] = binary_pattern
        
        return lbp

    def compute_texture(self, gray_img):
        try:
            features = []
            
            sobelx = cv.Sobel(gray_img, cv.CV_64F, 1, 0, ksize=3)
            sobely = cv.Sobel(gray_img, cv.CV_64F, 0, 1, ksize=3)
            
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            features.append(np.mean(gradient_magnitude))
            features.append(np.std(gradient_magnitude))
            
            hist, _ = np.histogram(gray_img, bins=256, range=(0, 256))
            hist = hist.astype(np.float32)
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log2(hist))
            features.append(entropy)
            
            features = [x if np.isfinite(x) else 0 for x in features]
            
            return np.array(features)
        except Exception as e:
            print(f"Error in simple texture features: {e}")
            return np.zeros(3)
    
    def extract_shape(self, img):
        try:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
            binary = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv.THRESH_BINARY, 11, 2)
            
            contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            contour_features = []
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
                    
                    contour_features.extend([area, perimeter, circularity, aspect_ratio])
            
            while len(contour_features) < 12:
                contour_features.append(0.0)
            
            return np.array(contour_features[:12])
        except Exception as e:
            print(f"Error in shape feature extraction: {e}")
            return np.zeros(12)
    
    def combine_features(self, img):
        try:
            deep_features = self.extract_deep(img)
            
            color_features = self.extract_moments(img)
            texture_features = self.extract_texture(img)
            shape_features = self.extract_shape(img)
            
            traditional_features = np.concatenate([
                color_features, 
                texture_features, 
                shape_features
            ])
            
            norm = np.linalg.norm(traditional_features)
            if norm > 0:
                traditional_features = traditional_features / norm
            
            alpha = 0.7
            combined_features = np.concatenate([
                alpha * deep_features,
                (1 - alpha) * traditional_features
            ])
            
            final_norm = np.linalg.norm(combined_features)
            if final_norm > 0:
                combined_features = combined_features / final_norm
            
            return combined_features.astype(np.float32)
            
        except Exception as e:
            print(f"Error building feature vector: {e}")
            return np.zeros(4096 + 6 + 60 + 12, dtype=np.float32)
    
    def compute_similarity(self, features1, features2):
        try:
            dot_product = np.dot(features1, features2)
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            
            if norm1 > 0 and norm2 > 0:
                similarity = dot_product / (norm1 * norm2)
                similarity = max(0.0, min(1.0, similarity))
            else:
                similarity = 0.0
                
            return similarity
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.0
    
    def load_or_extract_features(self, image_paths):
        if os.path.exists(feature_cache_file):
            print("Loading cached features...")
            try:
                with open(feature_cache_file, 'rb') as f:
                    self.database_features = pickle.load(f)
                print(f"Loaded {len(self.database_features)} features from cache.")
                return
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
            with open(feature_cache_file, 'wb') as f:
                pickle.dump(self.database_features, f)
            print(f"Features cached. Successfully processed {successful}/{len(image_paths)} images.")
        except Exception as e:
            print(f"Error caching features: {e}")

def retrieval(choice):
    categories = {
        '1': ('beach', 'image.query/beach.jpg'),
        '2': ('mountain', 'image.query/mountain.jpg'),
        '3': ('food', 'image.query/food.jpg'),
        '4': ('dinosaur', 'image.query/dinosaur.jpg'),
        '5': ('flower', 'image.query/flower.jpg'),
        '6': ('horse', 'image.query/horse.jpg'),
        '7': ('elephant', 'image.query/elephant.jpg')
    }
    
    print("Choose a category:")
    for key, (name, _) in categories.items():
        print(f"{key}: {name}")
    
    if choice not in categories:
        print("Invalid choice!")
        return
    
    category_name, query_path = categories[choice]
    
    retrieval_system = MediaRetrieval()
    
    query_img = retrieval_system.image_read(query_path)
    if query_img is None:
        print(f"Error loading query image: {query_path}")
        return
    
    print(f"Searching for: {category_name}")
    cv.imshow("Query Image", query_img)
    cv.waitKey(1)
    
    database_files = sorted(glob(os.path.join(database_dir, "*.jpg")))
    
    if not database_files:
        print("No images found in database!")
        return
    
    retrieval_system.load_or_extract_features(database_files)
    
    query_features = retrieval_system.combine_features(query_img)
    
    similarities = []
    for img_path, db_features in retrieval_system.database_features.items():
        similarity = retrieval_system.compute_similarity(query_features, db_features)
        img = retrieval_system.image_read(img_path)
        if img is not None:
            similarities.append((img_path, similarity, img))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    if similarities:
        best_match_path, best_score, best_img = similarities[0]
        cv.imshow("Best Match", best_img)
        print(f"\nBest match: {os.path.basename(best_match_path)}")
        print(f"Similarity score: {best_score:.4f}")
    
    print("Press any key to close windows...")
    cv.waitKey(0)
    cv.destroyAllWindows()

def main():
    print("Media Retrieval System")
    print("=" * 50)
    print("1: Image retrieval")
    print("2: Clear feature cache")
    
    try:
        choice = input("Choose an option: ").strip()
        
        if choice == '1':
            retrieval()
        elif choice == '2':
            if os.path.exists(feature_cache_file):
                os.remove(feature_cache_file)
                print("Feature cache cleared.")
            else:
                print("No cache file found.")
        else:
            print("Invalid choice!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()