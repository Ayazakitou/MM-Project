# retrieval_interface.py
from glob import glob
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QComboBox, 
                            QGroupBox, QFileDialog, QMessageBox, QProgressBar, 
                            QScrollArea, QGridLayout, QSplitter, QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont
import image_retrival_engine as ire
from categories import CATEGORIES

database_dir = "image.orig"
query_dir = "image.query"
feature_cache_file = "features_cache.pkl"

class SearchThread(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    
    def __init__(self, retrieval_system, query_path):
        super().__init__()
        self.retrieval_system = retrieval_system
        self.query_path = query_path
    
    def run(self):
        try:
            results = self.retrieval_system.search_similar(self.query_path, top_k=10)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

class ImageRetrievalInterface:
    def __init__(self):
        self.engine = ire.MediaRetrieval()

    def search_image(self, query_path: str, top_k: int = 5):
        return self.engine.search_image(query_path, top_k)
        
class ImageRetrievalFactory:
    @staticmethod
    def create_interface() -> ImageRetrievalInterface:
        return ImageRetrievalInterface()
    
    def __init__(self):
        pass

class RetrievalWorker(QWidget):

    finish = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, retrieval_system, top_k: int):
        super().__init__()
        self.retrieval_system = retrieval_system
        self.top_k = top_k
        self.init_ui()

    def run(self):
        try:
            results = self.retrieval_system.search_image(self.query_image_path, top_k=5)
            self.finish.emit(results)
        except Exception as e:
            self.error.emit(str(e))

class RetrievalApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.retrieval_system = ire.MediaRetrieval()
        self.init_ui()
        self.load_database()

    def init_ui(self):
        self.setWindowTitle("Image Retrieval System")
        self.setGeometry(200, 200, 1250, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        title = QLabel("Image Retrieval System")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)

        control_group = QGroupBox("Settings")
        control_layout = QVBoxLayout()

        category_layout = QHBoxLayout()
        category_layout.addWidget(QLabel("Choose the Category:"))
        
        self.category_combo = QComboBox()
        for key, (name, path) in CATEGORIES.items():
            self.category_combo.addItem(f"{key}. {name}", path)
        category_layout.addWidget(self.category_combo)
        
        self.search_btn = QPushButton("Start Search")
        self.search_btn.clicked.connect(self.start_search)
        category_layout.addWidget(self.search_btn)
        self.clear_btn = QPushButton("Clear Cache")
        self.clear_btn.clicked.connect(self.clear_cache)
        category_layout.addWidget(self.clear_btn)
        
        control_layout.addLayout(category_layout)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        result_group = QGroupBox("Check Results")
        result_layout = QVBoxLayout()

        self.query_label = QLabel()
        self.query_label.setAlignment(Qt.AlignCenter)
        self.query_label.setMinimumHeight(300)
        self.query_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        self.query_label.setText("Query Image Will Be Shown Here")
        result_layout.addWidget(QLabel("Query image:"))
        result_layout.addWidget(self.query_label)

        self.result_label = QLabel("Please select a category and click search.")
        self.result_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.result_label)

        self.results_grid = QGridLayout()
        result_widget = QWidget()
        result_widget.setLayout(self.results_grid)
        result_layout.addWidget(result_widget)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)

        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        
    def load_database(self):
        self.status_label.setText("Extracting/Loading Features...")
        QApplication.processEvents()
        database_files = sorted(glob(os.path.join(database_dir, "*.jpg")))

        if not database_files:
            print("No images found in database!")
            return
        
        success = self.retrieval_system.load_or_extract_features(database_files)
        if success:
            self.status_label.setText(f"Database Loaded. {len(self.retrieval_system.database_features)} images indexed.")
        else:
            self.status_label.setText("Failed to load database.")
            QMessageBox.critical(self, "Error", "Failed to load database features, please check the database directory.")
    
    def start_search(self):
        query_path = self.category_combo.currentData()
        if not query_path or not os.path.exists(query_path):
            QMessageBox.warning(self, "Error", "Invalid query image path.")
            return

        pixmap = QPixmap(query_path)
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.query_label.setPixmap(scaled_pixmap)
        
        self.status_label.setText("Finding...")
        self.search_btn.setEnabled(False)
        self.clear_results()

        self.search_thread = SearchThread(self.retrieval_system, query_path)
        self.search_thread.finished.connect(self.show_results)
        self.search_thread.error.connect(self.search_error)
        self.search_thread.start()

    def clear_cache(self):
        self.retrieval_system.clear_cache()
        self.status_label.setText("Feature cache cleared.")
    
    def show_results(self, results):
        self.search_btn.setEnabled(True)
        
        if not results:
            self.result_label.setText("Did not find any results.")
            self.status_label.setText("Finished-No results found.")
            return

        best_match = os.path.basename(results[0][0])
        best_score = results[0][1]
        self.result_label.setText(f"Finded {len(results)} results - Best Match: {best_match} (Similarity: {best_score:.3f})")

        row, col = 0, 0
        for i, (img_path, score) in enumerate(results[:10]):
            img_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setStyleSheet("border: 1px solid #ccc; margin: 2px;")
            
            pixmap = QPixmap(img_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(120, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                img_label.setPixmap(scaled_pixmap)

            info_label = QLabel(f"#{i+1}\n{score:.3f}")
            info_label.setAlignment(Qt.AlignCenter)
            info_label.setStyleSheet("font-size: 10px;")

            self.results_grid.addWidget(img_label, row*2, col)
            self.results_grid.addWidget(info_label, row*2+1, col)
            
            col += 1
            if col >= 5:
                col = 0
                row += 1
        
        self.status_label.setText(f"Finished-Finded {len(results)} results.")
    
    def clear_results(self):

        for i in reversed(range(self.results_grid.count())):
            widget = self.results_grid.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        self.result_label.setText("Please select a category and click search.")
    
    def search_error(self, error_msg):
        self.search_btn.setEnabled(True)
        self.status_label.setText("Search failed")
        QMessageBox.critical(self, "Error in searching", f"Error:\n{error_msg}")

def main():
    app = QApplication(sys.argv)
    window = RetrievalApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()